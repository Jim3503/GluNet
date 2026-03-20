import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

class SignIMG(nn.Module):
    def __init__(self, temperature=0.07, pos_threshold=0.7, neg_threshold=0.3, 
                 implicit_gloss_threshold=0.8, implicit_gloss_min_frames=3,
                 pos_samples=2, neg_samples=4):
        """
        初始化SignIMG模块，用于计算图像-文本之间的对比损失。

        Args:
            temperature (float): 温度参数，用于控制softmax的平滑度。
            pos_threshold (float): 正样本相似度阈值。
            neg_threshold (float): 负样本相似度阈值。
            implicit_gloss_threshold (float): 隐式gloss检测的相似度阈值。
            implicit_gloss_min_frames (int): 隐式gloss中最少包含的帧数。
            pos_samples (int): 每个样本最多选择的正样本数量。
            neg_samples (int): 每个样本最多选择的负样本数量。
        """
        super(SignIMG, self).__init__()
        self.temperature = temperature
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        self.implicit_gloss_threshold = implicit_gloss_threshold
        self.implicit_gloss_min_frames = implicit_gloss_min_frames
        self.pos_samples = pos_samples
        self.neg_samples = neg_samples
        
        # 初始化logit_scale并限制其最大值为log(100)，这有助于防止梯度爆炸
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
        self.max_logit_scale = np.log(100)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, image_features, text_features, alpha=0.5, beta=0.5, gamma=0.5):
        """
        计算图像-文本对比损失。

        Args:
            image_features: 图像特征 [batch_size, dim]
            text_features: 文本特征 [batch_size, dim]
        
        Returns:
            total_loss: 总损失
        """
        # 确保输入在同一设备上
        device = image_features.device
        
        # 检查输入特征是否有NaN值
        if torch.isnan(image_features).any() or torch.isnan(text_features).any():
            print("警告: 输入特征包含NaN值，返回零损失")
            return torch.tensor(0.0, device=device)
        
        # 确保text_features与image_features在同一设备上
        if text_features.device != device:
            text_features = text_features.to(device)
            
        # 检查输入特征的维度
        if len(image_features.shape) > 2 and image_features.shape[1] > 1:
            # 如果输入是帧序列特征 [batch_size, seq_len, dim]
            return self.forward_with_frames(image_features, text_features)
        
        # 特征归一化，添加小的epsilon防止除零
        image_features = F.normalize(image_features + 1e-8, dim=-1)
        text_features = F.normalize(text_features + 1e-8, dim=-1)
        
        batch_size = image_features.shape[0]
        
        # 批次大小检查，如果太小则返回零损失
        if batch_size <= 1:
            print("警告: 批次大小太小，无法计算对比损失")
            return torch.tensor(0.0, device=device)
        
        # 限制logit_scale的值，防止梯度爆炸
        self.logit_scale.data = torch.clamp(self.logit_scale.data, max=self.max_logit_scale)
        logit_scale = self.logit_scale.exp()
        
        # 计算单模态内的相似度矩阵
        image_sim = torch.matmul(image_features, image_features.t())
        text_sim = torch.matmul(text_features, text_features.t())
        
        # 检查相似度矩阵是否有NaN值
        if torch.isnan(image_sim).any() or torch.isnan(text_sim).any():
            print("警告: 相似度矩阵包含NaN值，返回零损失")
            return torch.tensor(0.0, device=device)
        
        # 计算跨模态相似度矩阵
        logits_per_image = logit_scale * torch.matmul(image_features, text_features.t())
        logits_per_text = logits_per_image.t()

        # 创建标签：对角线为正样本
        labels = torch.arange(batch_size, device=device)
        
        # 计算单模态对比损失
        image_loss = self._compute_modality_loss(image_sim)
        text_loss = self._compute_modality_loss(text_sim)
        
        # 计算跨模态对比损失
        try:
            cross_loss = (
                self.criterion(logits_per_image, labels) +
                self.criterion(logits_per_text, labels)
            ) / 2
            if torch.isnan(cross_loss) or torch.isinf(cross_loss):
                print("警告: 跨模态损失计算出现NaN/Inf，设置为0")
                cross_loss = torch.tensor(0.0, device=device)
        except Exception as e:
            print(f"计算跨模态损失出错: {e}")
            print(f"logits_per_image shape: {logits_per_image.shape}")
            print(f"logits_per_text shape: {logits_per_text.shape}")
            print(f"labels shape: {labels.shape}")
            cross_loss = torch.tensor(0.0, device=device)

        # 组合损失
        total_loss = alpha*image_loss + beta*text_loss + gamma*cross_loss  
        
        # 防止梯度爆炸
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"警告: 检测到无效损失值!")
            print(f"Image loss: {image_loss.item() if not torch.isnan(image_loss) and not torch.isinf(image_loss) else 'INVALID'}")
            print(f"Text loss: {text_loss.item() if not torch.isnan(text_loss) and not torch.isinf(text_loss) else 'INVALID'}")
            print(f"Cross loss: {cross_loss.item() if not torch.isnan(cross_loss) and not torch.isinf(cross_loss) else 'INVALID'}")
            return torch.tensor(0.0, device=device)
            
        return total_loss

    def _compute_modality_loss(self, sim_matrix):
        """
        计算单模态内的对比损失。
        
        Args:
            sim_matrix: 相似度矩阵 [batch_size, batch_size]
            
        Returns:
            loss: 对比损失
        """
        # 检查输入矩阵是否有效
        if torch.isnan(sim_matrix).any() or torch.isinf(sim_matrix).any():
            print("警告: 相似度矩阵包含NaN或Inf值，返回零损失")
            return torch.tensor(0.0, device=sim_matrix.device)
            
        batch_size = sim_matrix.size(0)
        
        # 移除自身相似度
        mask = torch.eye(batch_size, device=sim_matrix.device)
        sim_matrix = sim_matrix * (1 - mask)
        
        # 找到每个样本最相似的其他样本作为正样本
        # pos_mask = (sim_matrix > self.pos_threshold) * (1 - mask)
        pos_mask = (sim_matrix > self.pos_threshold) & (mask == 0)  # 布尔张量
        neg_mask = sim_matrix < self.neg_threshold
        
        # 如果没有找到正样本，使用最相似的样本
        if not pos_mask.any():
            # 获取每行（每个样本）中除自身外的最大值
            values, _ = torch.topk(sim_matrix * (1 - mask), k=min(2, batch_size), dim=1)
            min_pos_threshold = values[:, -1].min().item() - 1e-6  # 确保至少有一个正样本
            # pos_mask = (sim_matrix > min_pos_threshold) * (1 - mask)
            pos_mask = (sim_matrix > min_pos_threshold) & (mask == 0)  # 布尔张量
        
        loss = 0
        valid_samples = 0
        
        for i in range(batch_size):
            if pos_mask[i].any():
                # with open('debug.txt', 'a') as f:
                #     f.write(f"pos_mask[i]: {pos_mask[i]}/n")
                #     f.write(f"pos_mask[i]type: {type(pos_mask[i])}/n")
                pos_indices = torch.where(pos_mask[i])[0]
                # pos_logits = sim_matrix[i][pos_mask[i]]
                pos_logits = sim_matrix[i][pos_indices]
                    
                
                if neg_mask[i].any():
                    neg_indices = torch.where(neg_mask[i])[0]
                    neg_logits = sim_matrix[i][neg_indices]
                    
                    # 限制正负样本数量，防止内存溢出
                    pos_logits = pos_logits[:min(self.pos_samples, len(pos_logits))]
                    neg_logits = neg_logits[:min(self.neg_samples, len(neg_logits))]
                    
                    # 计算InfoNCE损失
                    logits = torch.cat([pos_logits.mean().unsqueeze(0), neg_logits])
                    scaled_logits = logits.unsqueeze(0)
                    target = torch.zeros(1, dtype=torch.long, device=logits.device)
                    
                    loss += F.cross_entropy(scaled_logits / self.temperature, target)
                    valid_samples += 1
        
        return loss / max(1, valid_samples)
    
    def forward_with_frames(self, frames_feature, text_feature):
        """
        使用帧级别特征计算损失，并发现隐式的gloss结构。
        
        Args:
            frames_feature: 视频帧特征 [batch_size, seq_len, dim]
            text_feature: 文本特征 [batch_size, dim] 或 [batch_size, seq_len, dim]
            
        Returns:
            total_loss: 总损失
        """
        batch_size = frames_feature.size(0)
        total_loss = 0
        
        # 归一化特征
        if len(text_feature.shape) == 2:
            # [batch_size, dim] -> [batch_size, 1, dim]
            text_feature = text_feature.unsqueeze(1)
            text_feature = F.normalize(text_feature, dim=-1)
        else:
            # [batch_size, seq_len, dim]
            text_feature = F.normalize(text_feature, dim=-1)
        
        # 对每个批次中的样本单独处理
        for b in range(batch_size):
            # 归一化当前样本的帧特征
            sample_frames = F.normalize(frames_feature[b], dim=-1)
            sample_text = text_feature[b]
            
            # 计算帧间相似度矩阵
            frame_similarities = torch.matmul(sample_frames, sample_frames.transpose(0, 1))
            
            # 发现隐式gloss（相似帧的聚类）
            implicit_glosses = []
            processed_frames = set()
            
            # 查找相似帧形成的gloss
            for i in range(len(sample_frames)):
                if i in processed_frames:
                    continue
                
                # 找到与当前帧相似度高于阈值的所有帧
                similar_frames = torch.where(frame_similarities[i] > self.implicit_gloss_threshold)[0].tolist()
                similar_frames = [f for f in similar_frames if f not in processed_frames]
                
                # 如果相似帧数量达到最小要求，认为这是一个隐式gloss
                if len(similar_frames) >= self.implicit_gloss_min_frames:
                    implicit_glosses.append(similar_frames)
                    processed_frames.update(similar_frames)
            
            # 如果找到了隐式gloss，计算基于gloss的损失
            if implicit_glosses:
                # 每个gloss的平均特征
                gloss_features = torch.stack([
                    sample_frames[gloss_frames].mean(dim=0)
                    for gloss_frames in implicit_glosses
                ])
                
                # 计算gloss与文本的相似度
                gloss_text_sim = torch.matmul(gloss_features, sample_text.transpose(0, 1))
                
                # 对每个gloss，找到最相似的文本token
                for g_idx, g_sim in enumerate(gloss_text_sim):
                    # 最相似的文本token作为正样本
                    pos_indices = torch.topk(g_sim, k=min(self.pos_samples, len(g_sim))).indices
                    
                    # 最不相似的文本token作为负样本
                    neg_indices = torch.topk(g_sim, k=min(self.neg_samples, len(g_sim)), largest=False).indices
                    
                    # InfoNCE损失: 让gloss特征与相应的文本token更相似
                    pos_logits = g_sim[pos_indices]
                    neg_logits = g_sim[neg_indices]
                    
                    logits = torch.cat([pos_logits.mean().unsqueeze(0), neg_logits])
                    labels = torch.zeros(len(logits), device=logits.device, dtype=torch.long)
                    
                    sample_loss = self.criterion(logits.unsqueeze(0) / self.temperature, labels.unsqueeze(0))
                    total_loss += sample_loss
            
            # 如果没有找到隐式gloss，退回到基本的跨模态损失
            else:
                # 平均所有帧特征
                avg_frame_feature = sample_frames.mean(dim=0, keepdim=True)
                
                # 计算与文本特征的相似度
                cross_sim = torch.matmul(avg_frame_feature, sample_text.transpose(0, 1))
                
                # 选择最相似的文本token作为正样本
                pos_indices = torch.topk(cross_sim[0], k=min(self.pos_samples, len(cross_sim[0]))).indices
                
                # 选择最不相似的文本token作为负样本
                neg_indices = torch.topk(cross_sim[0], k=min(self.neg_samples, len(cross_sim[0])), largest=False).indices
                
                # 计算InfoNCE损失
                pos_logits = cross_sim[0][pos_indices]
                neg_logits = cross_sim[0][neg_indices]
                
                logits = torch.cat([pos_logits.mean().unsqueeze(0), neg_logits])
                labels = torch.zeros(len(logits), device=logits.device, dtype=torch.long)
                
                sample_loss = self.criterion(logits.unsqueeze(0) / self.temperature, labels.unsqueeze(0))
                total_loss += sample_loss
        
        # 平均每个样本的损失
        total_loss = total_loss / batch_size
        
        # 防止梯度爆炸
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"警告: 帧级别损失计算中检测到无效值!")
            return torch.tensor(0.0, device=total_loss.device)
        
        return total_loss


if __name__ == "__main__":
    # Example usage of the SignCL class
    batch_size = 8
    seq_len_frames = 16
    seq_len_text = 16
    embed_dim = 64

    # Randomly generated video frame embeddings
    frames_feature = torch.randn(batch_size, seq_len_frames, embed_dim)

    # Randomly generated text token embeddings
    text_feature = torch.randn(batch_size, seq_len_text, embed_dim)

    # Initialize the SignCL model
    sign_img = SignIMG(max_distance=32.0, pos_samples=2, neg_samples=4, cross_modal_weight=0.5, intra_modal_weight=0.5)
    
    # Compute the contrastive loss
    loss = sign_img(frames_feature, text_feature)
    print(f"Contrastive Loss: {loss.item()}")
  
