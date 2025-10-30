import torch
import torch.nn as nn
from torch.nn import functional as F

from tqdm import tqdm
from copy import deepcopy
import numpy as np

from clip.clip import load, tokenize
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()

from .utils import build_cosine_scheduler, freeze_parameters
import pdb
import time
from .utils import get_context_indices
from torch.distributions.normal import Normal 
from torch.distributions.kl import kl_divergence
import time 

from .evaluator import Evaluator

class Adapter(nn.Module):
    def __init__(self, in_dim, out_dim, sigma=False, layer_num=1):
        super().__init__()

        self.fc = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.sigma = sigma
        # init_weights(self.fc)

    def forward(self, x):
        if self.sigma:
            return F.softplus(self.fc(x)) * 0.999 + 0.001
        else:
            return self.fc(x)
        
class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, x, tokenized_prompts):
        x = x + self.positional_embedding.type(self.dtype) #position_embeding可训练
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection # @ and
        return x


class CLIP(nn.Module):
    def __init__(self, args, class_names, clip_model,  vga, 
                 mu_adapters=None, sigma_adapters=None, task_tokens=None, 
                 task_to_cls_num=None, prompt_templates=None, previous_components=None,
                 task_to_distribution=None, mu_global_adapter=None, sigma_global_adapter=None,
                  global_vga=None):
        super().__init__()
        self.n_class = len(class_names)
        self.args = args
        # text enoder
        self.text_encoder = TextEncoder(clip_model)
        if torch.cuda.device_count() > 1:
            self.text_encoder = nn.DataParallel(self.text_encoder, device_ids=args.gpus)

        self.current_class_names = class_names
        # prompt learner
        ctx_dim = clip_model.ln_final.weight.shape[0]
        dtype = clip_model.dtype
        if previous_components is not None:
            self.unpack_prev_components(previous_components)

        # image encoder
        self.image_encoder = clip_model.visual
        self.vga = vga 
        self.vga_global = global_vga
        self.logit_scale = clip_model.logit_scale

        self.mu_adapters = mu_adapters
        self.sigma_adapters = sigma_adapters
        self.mu_global_adapter = mu_global_adapter
        self.sigma_global_adapter = sigma_global_adapter

        self.forward_times = self.args.forward_times
        self.forward_times_global = self.args.forward_times_global

        self.task_tokens = task_tokens
        self.task_to_cls_num = task_to_cls_num
        self.prompt_templates = prompt_templates
        self.pretrained_text_encoder = clip_model.encode_text
        self.prior_text_features()
        self.class_to_task_mapping = {} # for faster indexing to get task ids
        self.classwise_centroids = {}
        self.task_to_distribution = task_to_distribution
        self.init_new_heads()
        self.task_anchors = {}  # dict: task_id -> mean image embedding
    
    @torch.no_grad()
    def compute_task_anchor(self, task_id, dataloader):
        """Tính và lưu anchor cho task hiện tại"""
        all_embeddings = []
        for images, _, _ in dataloader:  # giả sử batch = (images, labels, index)
            imgs_emb = self.image_encoder(images.cuda(self.args.default_gpu)).detach()
            imgs_emb = imgs_emb / imgs_emb.norm(dim=-1, keepdim=True)
            all_embeddings.append(imgs_emb.cpu())
        all_embeddings = torch.cat(all_embeddings, dim=0)
        anchor = all_embeddings.mean(dim=0)
        self.task_anchors[task_id] = anchor

    def select_task_by_anchor(self, image):
        """Chọn task gần nhất với ảnh theo anchor"""
        img_emb = self.image_encoder(image.type(self.dtype))
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        best_task = None
        max_sim = -float('inf')
        for tid, anchor in self.task_anchors.items():
            anchor = anchor.to(img_emb.device)
            sim = F.cosine_similarity(img_emb, anchor.unsqueeze(0)).mean()  # anchor=[D] or [1,D]
            if sim > max_sim:
                max_sim = sim
                best_task = tid
        return best_task if best_task is not None else self.args.sess  # fallback if empty

    def forward(self, image, labels=None, test=False, finetuning=False, return_mean=True, for_prior=None):
        with torch.no_grad():
            image_features = self.image_encoder(image.type(self.dtype))
            image_features_normed = image_features / image_features.norm(dim=-1, keepdim=True)
            image_features = image_features_normed.detach()
            image_features_normed = image_features_normed.detach()

        n_class = self.n_class
        prev_cls_num = self.n_class - self.task_to_cls_num[self.args.sess]
        logit_scale = self.logit_scale.exp()
        if test:
            with torch.no_grad():
                text_features = self.frozen_text_features
                context = image_features_normed.clone() # torch.cat([image_features.unsqueeze(0), self.task_token_two[-1]], 1)
                n_query = text_features.shape[0]
                query = text_features.clone().unsqueeze(0)
                if self.args.expandable_tokens:
                    query = torch.cat([query] + [token for token in self.task_tokens], 1)                
                attn_mask = self.get_attention_mask((query.shape[1], query.shape[1]), self.args.sess+1, text_features.shape[0])
                if self.args.use_vga:
                    vga_features = self.vga(query, context.unsqueeze(0), tgt_mask=attn_mask).squeeze(0)
                
                rsamples_g = None 
                if self.args.hierarchical:
                    # vga_features_global = self.vga(query, context.unsqueeze(0)).squeeze(0)
                    global_input_features = vga_features[:n_query]  if self.args.use_vga else text_features
                    global_input_features = global_input_features + text_features
                    qdist_g = self.get_variational_adapter_features(global_input_features, global_adapter=True)
                    rsamples_g = qdist_g.rsample([self.forward_times_global])

                logits =[]
                samplewise_text_feats = []
                start_cls_idx, end_cls_idx = 0, 0
                for i in range(self.args.sess+1):
                    start_cls_idx = end_cls_idx
                    end_cls_idx += self.task_to_cls_num[i]
                    text_features_relevant = text_features[start_cls_idx:end_cls_idx].clone()
                    text_features_ = text_features_relevant
                    if self.args.use_vga:
                        text_features_ = text_features_ + vga_features[start_cls_idx:end_cls_idx] 
                    if self.args.expandable_tokens:
                        text_features_ = text_features_ + vga_features[n_query+i]

                    if self.args.hierarchical:
                        text_features_ = text_features_.unsqueeze(0).expand(self.forward_times_global, -1, -1) + rsamples_g[:, start_cls_idx:end_cls_idx, :]
                    qdist = self.get_variational_adapter_features(text_features_, i if self.args.expandable_adapter else 0)            
                    rsamples = qdist.rsample([self.forward_times])
                   
                    text_features_ = text_features_.unsqueeze(0).expand(self.forward_times, -1, -1, -1) if self.args.hierarchical else text_features_.unsqueeze(0).expand(self.forward_times, -1, -1)
                    if self.args.hierarchical:
                        rsamples = rsamples.flatten(0, 1)
                        text_features_ = text_features_.flatten(0, 1)
                    text_features_ = rsamples + text_features_ 
                    
                    logits_ = logit_scale * image_features_normed @ text_features_.permute(0, 2, 1) 
                  
                    logits.append(logits_)
                    if self.args.compute_ram:
                        samplewise_text_feats.append(text_features_relevant)
                # logits = torch.stack(logits, 0).sum(0)
                logits = torch.cat(logits, -1)
                logits = logits.detach()
            if self.args.compute_ram:
                visual_feats = image_features_normed
                samplewise_text_feats = torch.cat(samplewise_text_feats, 0)
                samplewise_text_feats = samplewise_text_feats / samplewise_text_feats.norm(dim=-1, keepdim=True)
                samplewise_text_feats = samplewise_text_feats[labels]
                return logits, (visual_feats.detach().cpu(), samplewise_text_feats.detach().cpu())
            if return_mean:
                return logits.mean(0), (None, None)
            else:
                return logits, (None,None)

        else:
            
            text_features = self.frozen_text_features
            logits =[]
            kl_losses = []
            prior_matching_losses = []
            start_cls_idx, end_cls_idx = 0, 0
            context = image_features_normed.clone() 
            n_query = text_features.shape[0]
            query = text_features.clone().unsqueeze(0)
            if self.args.expandable_tokens:
                query = torch.cat([query] + [token for token in self.task_tokens], 1)
            attn_mask = self.get_attention_mask((query.shape[1], query.shape[1]), self.args.sess+1, text_features.shape[0])
            if self.args.use_vga:
                vga_features_all = self.vga(query, context.unsqueeze(0), tgt_mask=attn_mask).squeeze(0)
            
            rsamples_g = None 
            if self.args.hierarchical:
                # vga_features_global = self.vga(query, context.unsqueeze(0)).squeeze(0)
                global_input_features = vga_features_all[:n_query] if self.args.use_vga else text_features
                global_input_features = global_input_features + text_features
                pdist_g = self.get_prior_dist(context, global_input_features, labels, self.args.sess+1, 
                                                None, 
                                                None,
                                                use_np_prior=self.args.use_np_prior if not finetuning else False,
                                                global_adapter=True
                                                )
                qdist_g = self.get_variational_adapter_features(global_input_features, global_adapter=True)
                # pdist_g = self.get_prior_dist(text_features=global_input_features, use_np_prior=False)
                prior_matching_losses.append(kl_divergence(qdist_g, pdist_g).mean(0).sum() * 0.001)
                rsamples_g = qdist_g.rsample([self.forward_times_global])
                if self.args.lasp  and self.args.beta > 0:
                    prior_text_features = self.frozen_text_features_individual.clone()
                    sims = torch.stack([prior_text_features @ rsamples_g[r].t() for r in range(rsamples_g.shape[0])], 0)
                    sims = sims.mean(2).mean(0)
                    kl_losses.append(F.cross_entropy(sims,  torch.arange(sims.size(0)).cuda(device=self.args.default_gpu)) * self.args.beta)


            if self.args.distill and self.args.sess > 0 and self.args.alpha > 0:
                with torch.no_grad():
                    prev_task_text_features = text_features[:-self.task_to_cls_num[self.args.sess]].clone()
                    n_query_prev = prev_task_text_features.shape[0]
                    prev_vga_query = prev_task_text_features.unsqueeze(0)
                    if self.args.expandable_tokens:
                        prev_vga_query = torch.cat([prev_vga_query] + [token for token in self.previous_task_tokens], 1)
                    prev_attn_mask = self.get_attention_mask((prev_vga_query.shape[1], prev_vga_query.shape[1]), self.args.sess, prev_task_text_features.shape[0])
                    prev_vga_features_all = self.previous_vga(prev_vga_query, context.unsqueeze(0), tgt_mask=prev_attn_mask).squeeze(0).detach()
                    prev_global_input_features = prev_vga_features_all[:n_query_prev] + prev_task_text_features
                    qdist_g_prev = self.get_variational_adapter_features(prev_global_input_features, distill=True, global_adapter=True)
                    prev_loc = qdist_g_prev.loc.detach()
                kl_losses.append(F.mse_loss(prev_loc, qdist_g.loc[:prev_loc.shape[0]]) * 0.3)

            per_sample_text_feats = []
            taskwise_means = []

            for i in range(self.args.sess+1):
                
                start_cls_idx = end_cls_idx
                end_cls_idx += self.task_to_cls_num[i]
                if start_cls_idx not in self.class_to_task_mapping:
                    # update class to task mapping for faster indexing of task id based on class label id
                    self.class_to_task_mapping.update(dict(zip(np.arange(start_cls_idx, end_cls_idx), [i] * (end_cls_idx - start_cls_idx))))

                text_features_relevant = text_features.clone()[start_cls_idx:end_cls_idx]
                if self.args.use_vga:
                    vga_features = vga_features_all[start_cls_idx:end_cls_idx]
                    if self.args.expandable_tokens:
                        vga_features = vga_features + vga_features_all[n_query+i]
                    text_features_ = text_features_relevant + vga_features
                else:
                    text_features_ = text_features_relevant

                if self.args.hierarchical:
                    text_features_ = text_features_.unsqueeze(0).expand(self.forward_times_global, -1, -1) + rsamples_g[:, start_cls_idx:end_cls_idx, :]
                qdist = self.get_variational_adapter_features(text_features_, i if self.args.expandable_adapter else 0)            
                rsamples = qdist.rsample([self.forward_times])
                
                text_features_ = text_features_.unsqueeze(0).expand(self.forward_times, -1, -1, -1) if self.args.hierarchical else text_features_.unsqueeze(0).expand(self.forward_times, -1, -1)
                if self.args.hierarchical:
                    rsamples = rsamples.flatten(0, 1)
                    text_features_ = text_features_.flatten(0, 1)
                text_features_ = rsamples + text_features_ 
                
                taskwise_means.append(rsamples.mean(0))
                if self.args.lasp  and self.args.beta > 0 and (finetuning or (not finetuning and  self.args.sess == i)):
                    prior_text_features = self.frozen_text_features_individual.clone()[start_cls_idx:end_cls_idx]
                    sims = torch.stack([prior_text_features @ rsamples[r].t() for r in range(rsamples.shape[0])], 0)
                    sims = sims.mean(2).mean(0)
                    kl_losses.append(F.cross_entropy(sims,  torch.arange(sims.size(0)).cuda(device=self.args.default_gpu)) * self.args.beta)
                logits_ = (logit_scale * image_features_normed @ text_features_.permute(0, 2, 1)) 
                if finetuning or (not finetuning and self.args.sess == i):
                    if self.args.frozen_prior:
                        prior_text_features = self.frozen_text_features_individual.clone()[start_cls_idx:end_cls_idx] 
                        pdist = self.get_variational_adapter_features(prior_text_features.mean(1), i if self.args.expandable_adapter else 0)
                    else:
                        pdist = self.get_prior_dist(context, text_features_relevant, labels, i, 
                                                None, 
                                                self.task_tokens[i] if self.args.expandable_tokens else None,
                                                use_np_prior=self.args.use_np_prior, #if not finetuning else False,
                                                tgt_mask=attn_mask
                                                )
                    prior_matching_losses.append(kl_divergence(qdist, pdist).mean(0).sum() * 0.001)    
                
                logits.append(logits_)
                if (self.args.get_interclass_dist and self.args.sess == 9 and finetuning) or (self.args.get_adapter_distances and self.args.sess > 0):
                    with torch.no_grad():                        
                        per_sample_text_feats.append(rsamples.clone().detach().mean(0))
            
           
            if self.args.ortho_loss and self.args.sess >= 0:
                taskwise_means = torch.cat(taskwise_means)
                # taskwise_means = taskwise_means / taskwise_means.norm(dim=-1, keepdim=True)
                sims = taskwise_means @ taskwise_means.t()
                kl_losses.append(F.cross_entropy(sims,  torch.arange(sims.size(0)).cuda(device=self.args.default_gpu)) * 5)
                
            logits = torch.cat(logits, -1)
           
            kl_loss = sum(kl_losses)  if len(kl_losses) else 0.
            prior_matching_loss = sum(prior_matching_losses) 
            # prior_matching_loss = prior_matching_loss * 0.01 #if not finetuning else prior_matching_loss * 0.1 
            
            avg_cos_distance = None
            if self.args.get_adapter_distances and self.args.sess > 0:
                with torch.no_grad():
                    per_sample_text_feats_ = torch.stack(per_sample_text_feats, 0)
                    avg_cos_distance = self.get_avg_inter_adapter_distance(per_sample_text_feats_)
                    

            if self.args.get_interclass_dist and self.args.sess == 9 and finetuning:
                with torch.no_grad():
                    per_sample_text_feats_ = torch.cat(per_sample_text_feats, 0)
                    for label in np.arange(per_sample_text_feats_.shape[0]):
                        if label not in self.classwise_centroids:
                            self.classwise_centroids[label] = per_sample_text_feats_[label].unsqueeze(0)
                        else:
                            self.classwise_centroids[label] = torch.cat([self.classwise_centroids[label], per_sample_text_feats_[label].unsqueeze(0)], 0)
                  
            return logits, (kl_loss, prior_matching_loss, avg_cos_distance)

    def get_kld_loss(self, logits, logits_prior):
        student_conf = -torch.logsumexp(logits, dim=-1)
        teacher_conf = -torch.logsumexp(logits_prior, dim=-1)
        # if confidence > 1, it means student has a higher energy in which case the instance should be distilled using teacher logits
        confidence_ratio = student_conf / teacher_conf 
        mask = confidence_ratio > 1
        student_dist = F.log_softmax(logits[mask], dim=-1)
        teacher_dist = F.softmax(logits_prior[mask], dim=-1)
        # kld = -1. * (student_dist * teacher_dist).sum(-1).mean()#.unsqueeze(0).expand(student_dist.shape[0], -1, -1)).sum(-1).mean()    
        kld =  nn.KLDivLoss(reduction='batchmean')(student_dist, teacher_dist)#.sum(-1).mean()     
        return kld * 0.1
        
    def get_naive_distillation_loss(self, curr_model_logits, image_feats, image_feats_normed, prev_cls_num):
        # from the BiC paper (Large scale incremental learning)
        with torch.no_grad():
            prev_model_logits = self.forward_prev_model(image_feats, image_feats_normed)
            prev_model_logits = prev_model_logits.detach()

        kl_loss = nn.KLDivLoss(reduction='none')(F.log_softmax(curr_model_logits[:, :, :prev_cls_num], dim=-1), 
                                                 F.softmax(prev_model_logits, dim=-1)).sum(-1).mean()
        lamb = prev_cls_num / self.n_class
        return kl_loss * lamb

    @torch.no_grad()
    def set_classifier(self):
        pass 

    @property #变成属性
    def dtype(self):
        return self.image_encoder.conv1.weight.dtype #return int/float


class ClClipVariational(Evaluator):
    def __init__(self, args, use_float32=False, use_grad_checkpoint=False):
        super().__init__(args)
        self.args = args
        clip_model, _ = load(args.ckpt_path, device=f"cuda:{args.default_gpu}")
        clip_model.eval()
        if use_float32:
            clip_model.float()
        self.clip_model = clip_model
        self.use_grad_checkpoint = use_grad_checkpoint
        ctx_dim = self.clip_model.ln_final.weight.shape[0]

        self.lr = args.lr*args.train_batch/20
        self.wd = args.wd # wd ??
        self.epochs = args.epochs
        self.train_batch = args.train_batch 
        self.args = args
        self.current_class_names = []
        decoder_layer = torch.nn.TransformerDecoderLayer(d_model=ctx_dim, nhead=1, activation='gelu', batch_first=True).cuda(device=self.args.default_gpu).type(self.clip_model.dtype)
        self.vga = torch.nn.TransformerDecoder(decoder_layer, 1) if self.args.use_vga else None
        
        self.get_variational_adapters(ctx_dim)
        self.vga_global = None 
        if self.args.hierarchical:
            self.get_variational_adapters(ctx_dim, global_adapter=True)


        self.init_task_tokens(ctx_dim)
        
        self.task_to_cls_num = {}
        self.task_to_distribution = {}

        # for distillation
        self.previous_mu_adapters, self.previous_mu_global_adapter = None, None
        self.previous_sigma_adapters, self.previous_sigma_global_adapter = None, None
        self.previous_task_tokens = None
        self.previous_vga = None

    def init_task_tokens(self, ctx_dim):
        task_token = torch.zeros((1, 1,  ctx_dim), dtype=self.clip_model.dtype, requires_grad=True).cuda(device=self.args.default_gpu) 
        nn.init.normal_(task_token, std=.02)
        self.task_tokens =  nn.ParameterList([nn.Parameter(task_token)]) if self.args.expandable_tokens else None 

    @staticmethod
    def get_div_logits(outputs, nb_old_classes, nb_new_classes):
        outputs_div = outputs[:, :, nb_old_classes:nb_old_classes+nb_new_classes] 
        outputs_old = outputs[:, :, :nb_old_classes].max(-1)[0].unsqueeze(-1)
        outputs_div = torch.cat([outputs_old, outputs_div],  -1)
        return outputs_div
    
    def get_div_loss(self, outputs_div, div_targets):
        nb_old_classes = sum([self.task_to_cls_num[t_num] for t_num in range(self.args.sess)])

        mask_old_cls = div_targets < nb_old_classes
        mask_new_cls = ~mask_old_cls
        div_targets[mask_old_cls] = 0
        div_targets[mask_new_cls] -= nb_old_classes - 1
        aux_loss = F.cross_entropy(outputs_div.view(-1, outputs_div.shape[-1]), 
                                       div_targets) 
        return aux_loss

    def get_variational_adapters(self, ctx_dim, global_adapter=False):
        if not global_adapter:
            self.mu_adapters = nn.ModuleList([Adapter(ctx_dim, ctx_dim).cuda(device=self.args.default_gpu).type(self.clip_model.dtype)])
            self.sigma_adapters = nn.ModuleList([Adapter(ctx_dim, ctx_dim, sigma=True).cuda(device=self.args.default_gpu).type(self.clip_model.dtype)])
            self.mu_adapter_deter = None
        else:
            self.mu_global_adapter = Adapter(ctx_dim, ctx_dim).cuda(device=self.args.default_gpu).type(self.clip_model.dtype)
            self.sigma_global_adapter = Adapter(ctx_dim, ctx_dim, sigma=True).cuda(device=self.args.default_gpu).type(self.clip_model.dtype)

    def fit(self, data):
        self.task_to_cls_num[self.args.sess] = len(data['class_names'])
        self.current_class_names += data['class_names']
        print(f"Classes: {self.current_class_names}")
        train_loader = data['train_loader']

        if len(train_loader.dataset)< self.train_batch:
            real_img_bsz = len(train_loader.dataset)
            self.lr = self.lr * real_img_bsz / self.train_batch 
        else:
            real_img_bsz = self.train_batch

        per_epoch_steps = len(train_loader)

        self.init_model(class_names=self.current_class_names, per_epoch_steps=per_epoch_steps, prompt_templates=data['prompt_templates'])

        inter_adapter_distances = []
        run_times = []
        # self.model.eval()
        if self.model.vga is not None:
            self.model.vga.train()
        if self.args.sess >= 0:
            for epoch in tqdm(range(self.epochs)):
                for idx, (x, y, index) in tqdm(enumerate(train_loader), total=len(train_loader), desc = 'Training'):

                    cur_iter_idx = epoch*per_epoch_steps+idx
                    self.cur_iter_idx = cur_iter_idx
                    self.scheduler.step(cur_iter_idx)
                    start_time = time.time()
                    output, (kl_loss, prior_matching_loss, inter_adapter_distance) = self.model(x.cuda(device=self.args.default_gpu), y)
                    run_time = time.time() - start_time
                    run_times.append(run_time)
                    y = y.cuda(device=self.args.default_gpu)
                    loss = 0.
                    # pdb.set_trace()
                    if self.args.variational:
                        targets = y.unsqueeze(0).expand(output.shape[0], -1).contiguous().view(-1)
                        # if self.args.sess > 0:
                        #     loss = loss + self.get_div_loss(output.clone(), targets.clone()) * 0.01
                        output = output.view(-1, output.shape[-1])
                    else:
                        targets = y 
                    loss = loss + F.cross_entropy(output, targets) + kl_loss + prior_matching_loss
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    if inter_adapter_distance is not None and (epoch == self.epochs-1):
                        inter_adapter_distances.append(inter_adapter_distance)

            if self.args.sess > 0 and self.args.expandable_tokens:
                self.epoch_log()
            if len(inter_adapter_distances):
                print(f"Average inter-adapter distance: {np.mean(inter_adapter_distance)}")
        # if self.args.sess == 9 and self.args.get_interclass_dist:
        #     with torch.no_grad():
        #         self.compute_class_centroids()

        # pdb.set_trace()
            # print(self.model.image_encoder.layer1[0].conv1.weight[0])
        print(f"Average run time: {np.mean(run_times)}")
        self.model.eval()
        
        self.model.compute_task_anchor(self.args.sess, train_loader)
        if self.model.vga is not None:
            self.model.vga.train()
        return self.model

    @torch.no_grad()
    def compute_class_centroids(self):
        class_embeddings = {}
        for cls,  class_embedding in self.model.classwise_centroids.items():
            class_embeddings[cls] = class_embedding.mean(0)
        class_embeddings = dict(sorted(class_embeddings.items()))
        class_embeddings = torch.stack(list(class_embeddings.values()))
        class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
        pairwise_cosine_dists = class_embeddings @ class_embeddings.t()
        pairwise_cosine_dists = pairwise_cosine_dists.cpu()
        torch.save(pairwise_cosine_dists, "3.pt")

    def post_training(self, finalize=False):
        self.model.eval()
        self.model.set_classifier()
        if self.args.distill and finalize:
            self.preserve_copy_for_distillation()

    def finetuning(self, data):
        self.unfreeze_for_finetuning()
        self.cur_iter_idx = 0
        memory_loader = data['memory_loader']
        if len(memory_loader.dataset)< self.train_batch:
            real_img_bsz = len(memory_loader.dataset)
            self.lr = self.lr * real_img_bsz / self.train_batch 
        else:
            real_img_bsz = self.train_batch
            
        per_epoch_steps=len(memory_loader)
        inter_adapter_distances = []
        self.build_optimizer(per_epoch_steps=per_epoch_steps, lr=self.lr/10., warmup=False, finetune=True)
        if self.model.vga is not None:
            self.model.vga.eval()
        for epoch in tqdm(range(self.args.finetune_epochs)):
            for idx, (x, y, index) in tqdm(enumerate(memory_loader), total=len(memory_loader), desc = 'Finetuning'):

                cur_iter_idx = epoch*per_epoch_steps+idx
                self.cur_iter_idx = cur_iter_idx
                self.scheduler.step(cur_iter_idx)

                output, (kl_loss, prior_matching_loss, inter_adapter_distance) = self.model(x.cuda(device=self.args.default_gpu), y, finetuning=True)
                # pdb.set_trace()
                y = y.cuda(device=self.args.default_gpu)
                # pdb.set_trace()
                loss = 0.
                if self.args.variational:
                    targets = y.unsqueeze(0).expand(output.shape[0], -1).contiguous().view(-1)
                    output = output.view(-1, output.shape[-1])
                else:
                    targets = y 
                loss = loss + F.cross_entropy(output, targets) + kl_loss + prior_matching_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if inter_adapter_distance is not None and (epoch == self.epochs-1):
                        inter_adapter_distances.append(inter_adapter_distance)
        if self.args.sess == 9 and self.args.get_interclass_dist:
            with torch.no_grad():
                self.compute_class_centroids()
        if len(inter_adapter_distances):
                print(f"Average inter-adapter distance: {np.mean(inter_adapter_distance)}")

        if self.args.sess > 0 and self.args.expandable_tokens:
            self.epoch_log()
        
    @torch.no_grad()
    def preserve_copy_for_distillation(self):
        self.model.eval()
        self.previous_mu_adapters = deepcopy(self.model.mu_adapters)
        self.previous_sigma_adapters = deepcopy(self.model.sigma_adapters)
        self.previous_task_tokens = deepcopy(self.model.task_tokens)
        self.previous_vga = deepcopy(self.model.vga)
        if self.args.hierarchical:
            self.previous_mu_global_adapter = deepcopy(self.model.mu_global_adapter)
            self.previous_sigma_global_adapter = deepcopy(self.model.sigma_global_adapter)
            freeze_parameters(self.previous_mu_global_adapter, requires_grad=False)
            freeze_parameters(self.previous_sigma_global_adapter, requires_grad=False)
        freeze_parameters(self.previous_mu_adapters, requires_grad=False)
        freeze_parameters(self.previous_sigma_adapters, requires_grad=False)
        freeze_parameters(self.previous_task_tokens, requires_grad=False)
        freeze_parameters(self.previous_vga, requires_grad=False)

    def expand_task_token_list(self):
        new_task_token = deepcopy(self.task_tokens[-1])
        nn.init.trunc_normal_(new_task_token, std=.02)
        self.task_tokens.append(new_task_token)
        freeze_parameters(self.task_tokens[:-1], requires_grad=False)
        freeze_parameters(self.task_tokens[-1], requires_grad=True)

    def expand_adapter(self):
        ctx_dim = self.clip_model.ln_final.weight.shape[0]
        dtype = self.clip_model.dtype
        new_mu = Adapter(ctx_dim, ctx_dim).cuda(device=self.args.default_gpu).type(dtype)
        new_sigma = Adapter(ctx_dim, ctx_dim, sigma=True).cuda(device=self.args.default_gpu).type(dtype)
        self.mu_adapters.append(new_mu)
        self.sigma_adapters.append(new_sigma)
        self.mu_adapters[:-1].eval()
        self.sigma_adapters[:-1].eval()
        freeze_parameters(self.mu_adapters[:-1], requires_grad=False)
        freeze_parameters(self.sigma_adapters[:-1], requires_grad=False)
        freeze_parameters(self.mu_adapters[-1], requires_grad=True)
        freeze_parameters(self.sigma_adapters[-1], requires_grad=True)
        
    def unfreeze_for_finetuning(self, requires_grad=True):
        freeze_parameters(self.vga, requires_grad=False)
        freeze_parameters(self.mu_adapters[:-1], requires_grad=requires_grad)
        freeze_parameters(self.sigma_adapters[:-1], requires_grad=requires_grad)
        if self.args.expandable_tokens:
            freeze_parameters(self.task_tokens[:-1], requires_grad=requires_grad)
        if requires_grad:
            self.mu_adapters[:-1].train()
            self.sigma_adapters[:-1].train()
    
    def init_model(self, class_names, per_epoch_steps, prompt_templates=None):
        if self.args.sess > 0:
            freeze_parameters(self.vga, requires_grad=True)
            if self.args.expandable_tokens:
                self.expand_task_token_list()
            if self.args.expandable_adapter:
                self.expand_adapter()
            if self.args.expandable_prompt:
                self.expand_prompts()

        self.n_class = len(class_names)
        clip_model = deepcopy(self.clip_model)

        prev_model_components = (
                                 self.previous_mu_adapters, self.previous_sigma_adapters, 
                                 self.previous_task_tokens, self.previous_vga, 
                                 self.previous_mu_global_adapter, self.previous_sigma_global_adapter )
        self.model = CLIP(self.args, class_names, clip_model, self.vga,  
                          mu_adapters=self.mu_adapters, sigma_adapters=self.sigma_adapters,
                          task_tokens=self.task_tokens, task_to_cls_num = self.task_to_cls_num,
                          prompt_templates=prompt_templates, previous_components=prev_model_components,
                          task_to_distribution=self.task_to_distribution,
                          mu_global_adapter=self.mu_global_adapter if self.args.hierarchical else None, 
                          sigma_global_adapter=self.sigma_global_adapter if self.args.hierarchical else None,
                           global_vga=self.vga_global
                          )
        self.model.eval()
        if self.use_grad_checkpoint:
            try:
                self.model.text_encoder.transformer.use_gradient_checkpoint = True 
            except:
                self.model.text_encoder.module.transformer.use_gradient_checkpoint = True


        self.build_optimizer(per_epoch_steps, lr=self.lr, warmup=True)
       

    def build_optimizer(self, per_epoch_steps, lr, warmup=False, finetune=False):
        for name, param in self.model.named_parameters():
            if "vga" not in name and "task_token" not in name and "adapter" not in name:
                param.requires_grad_(False)
            
        # double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)

        print(f"\nParameters to be updated: {sorted(enabled)}\n")

        param_dict = [{'params': [p for p in self.model.parameters() if p.requires_grad]}]

        self.optimizer = torch.optim.SGD(param_dict, lr=lr, weight_decay=self.wd)
        total_step=self.epochs*per_epoch_steps if not finetune else self.args.finetune_epochs*per_epoch_steps
        warmup_steps = int(0.3 * total_step) if warmup else 0
        self.scheduler = build_cosine_scheduler(
            self.optimizer,
            lr=lr,
            total_step=total_step,
            lr_warmup_step=warmup_steps
            )
        
    @torch.no_grad()
    def inference(self,image, label, num_test, test_class):
        self.model.eval()
        logits, feats = self.model(image, label, test=True, return_mean=False)
        return logits.float(), feats

    
    @torch.no_grad()
    def epoch_log(self):
        """Write here whatever you want to log on the internal state of the model."""
        log = {}

        # Compute mean distance between class tokens
        mean_dist, min_dist, max_dist = [], float('inf'), 0.
        for i in range(len(self.task_tokens)):
            for j in range(i + 1, len(self.task_tokens)):
                dist = torch.norm(self.task_tokens[i] - self.task_tokens[j], p=2).item()
                mean_dist.append(dist)

                min_dist = min(dist, min_dist)
                max_dist = max(dist, max_dist)

        if len(mean_dist) > 0:
            mean_dist = sum(mean_dist) / len(mean_dist)
        else:
            mean_dist = 0.
            min_dist = 0.

        assert min_dist <= mean_dist <= max_dist, (min_dist, mean_dist, max_dist)
        log['token_mean_dist'] = round(mean_dist, 5)
        log['token_min_dist'] = round(min_dist, 5)
        log['token_max_dist'] = round(max_dist, 5)
        print(f"\n{log}")
