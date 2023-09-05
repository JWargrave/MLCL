import torch
from torch import nn
import torch.nn.functional as F


class Combiner(nn.Module):
    def __init__(self, clip_feature_dim: int, projection_dim: int, hidden_dim: int,learn_t=0):
        super(Combiner, self).__init__()
        self.text_projection_layer = nn.Linear(clip_feature_dim, projection_dim)
        self.image_projection_layer = nn.Linear(clip_feature_dim, projection_dim)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

        self.combiner_layer = nn.Linear(projection_dim * 2, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, clip_feature_dim)

        self.dropout3 = nn.Dropout(0.5)
        self.dynamic_scalar = nn.Sequential(nn.Linear(projection_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(0.5),
                                            nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.dropout4=nn.Dropout(0.0)
        self.combiner_proj=nn.Linear(hidden_dim,clip_feature_dim)


        if learn_t==1:
            self.logit_scale = nn.Parameter(torch.log(torch.FloatTensor((100,))))
            self.logit_scale_sl=nn.Parameter(torch.log(torch.FloatTensor((100,))))
        elif learn_t==0:
            self.logit_scale = 100
            self.logit_scale_sl=100
        print('Combiner using learnable temperature:',learn_t)
        self.learn_t=learn_t

        self.loss_weight=nn.Parameter(torch.log(torch.FloatTensor((0.5,))))

    def forward(self, image_features: torch.tensor, text_features: torch.tensor,
                target_features: torch.tensor,image_features_aug:torch.tensor) -> torch.tensor:
        
        predicted_features_1,predicted_features_2,norm_sum_feature,mid_raw_cross_modal_logits,cross_modal_logits_1 = self.combine_features_train_image_aug(image_features, text_features,image_features_aug)
        target_features = F.normalize(target_features, dim=-1)
        logit_scale=self.logit_scale.exp() if self.learn_t==1 else self.logit_scale
        logits1 = logit_scale * predicted_features_1 @ target_features.T
        logits2 = logit_scale * predicted_features_2 @ target_features.T
        logits3 = logit_scale * norm_sum_feature @ target_features.T
        return logits1,logits2,logits3,mid_raw_cross_modal_logits,cross_modal_logits_1

    def combine_features_train_image_aug(self, image_features: torch.tensor, text_features: torch.tensor,image_features_aug:torch.tensor):

        norm_sum_feature=F.normalize(image_features+text_features,dim=-1)

        logit_scale=self.logit_scale_sl.exp() if self.learn_t==1 else self.logit_scale_sl

        text_projected_features = self.dropout1(F.relu(self.text_projection_layer(text_features)))
        image_projected_features = self.dropout2(F.relu(self.image_projection_layer(image_features)))
        image_projected_features_aug=self.dropout2(F.relu(self.image_projection_layer(image_features_aug)))

        raw_combined_features = torch.cat((text_projected_features, image_projected_features), -1)
        raw_combined_features_aug = torch.cat((text_projected_features, image_projected_features_aug), -1)

        mid_raw_cross_modal_logits=logit_scale*F.normalize(raw_combined_features,dim=-1)@F.normalize(raw_combined_features_aug,dim=-1).T
        
        combined_features = self.dropout3(F.relu(self.combiner_layer(raw_combined_features)))
        combined_features_aug=self.dropout3(F.relu(self.combiner_layer(raw_combined_features_aug)))

        mix_feature=self.combiner_proj(self.dropout4(F.relu(combined_features)))
        norm_mix_feature=F.normalize(mix_feature,dim=-1)
        dynamic_scalar = self.dynamic_scalar(raw_combined_features)
        dynamic_scalar_aug = self.dynamic_scalar(raw_combined_features_aug)

        convex_comb=dynamic_scalar * text_features + (1 - dynamic_scalar) * image_features
        convex_comb_aug=dynamic_scalar_aug * text_features + (1 - dynamic_scalar_aug) * image_features_aug
        

        output = self.output_layer(combined_features) + convex_comb
        norm_output=F.normalize(output,dim=-1)
        output_aug=self.output_layer(combined_features_aug) + convex_comb_aug
        norm_output_aug=F.normalize(output_aug,dim=-1)

        cross_modal_logits_1=logit_scale*norm_output@norm_output_aug.T

        return (norm_output,\
                norm_mix_feature,\
                norm_sum_feature,\
                mid_raw_cross_modal_logits,\
                cross_modal_logits_1)
    
    def combine_features_weighted(self, image_features: torch.tensor, text_features: torch.tensor):
        norm_sum_feature=F.normalize(image_features+text_features,dim=-1)

        text_projected_features = self.dropout1(F.relu(self.text_projection_layer(text_features)))
        image_projected_features = self.dropout2(F.relu(self.image_projection_layer(image_features)))

        raw_combined_features = torch.cat((text_projected_features, image_projected_features), -1)
        combined_features = self.dropout3(F.relu(self.combiner_layer(raw_combined_features)))

        mix_feature=self.combiner_proj(self.dropout4(F.relu(combined_features)))

        dynamic_scalar = self.dynamic_scalar(raw_combined_features)
        convex_comb=dynamic_scalar * text_features + (1 - dynamic_scalar) * image_features
        

        output = self.output_layer(combined_features) + convex_comb
        return ((2-2*torch.sigmoid(self.loss_weight))*F.normalize(output, dim=-1),2*torch.sigmoid(self.loss_weight)*F.normalize(mix_feature,dim=-1),norm_sum_feature)
    
