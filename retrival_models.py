import torch
import torch.nn as nn

class EmbedBranch(nn.Module):
    '''
    Create a fully connected layer with L2 regularization to match dimensions of feature embedding for fusion
    :param feat_dim: input dimension of features previously extracted
    :param embedding_dim: output dimension of fetures embedding to give to fusion model
    '''
    def __init__(self, feat_dim, embedding_dim):
        super(EmbedBranch, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(feat_dim, embedding_dim),
                        nn.BatchNorm1d(embedding_dim),
                        nn.ReLU(inplace=True),
                        nn.Dropout(p=0.5)).cuda()

    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.normalize(x)
        return x

class EmbedBranch_Video(nn.Module):
    '''
    Create a fully connected layer with L2 regularization to match dimensions of feature embedding for fusion
    :param feat_dim: input dimension of features previously extracted
    :param embedding_dim: output dimension of fetures embedding to give to fusion model
    '''
    def __init__(self, embedding_dim):
        super(EmbedBranch_Video, self).__init__()
        self.conv1=nn.Conv2d(1, 1,kernel_size=10)
        self.lin=nn.Linear(1015, embedding_dim)

        self.fc1 = nn.Sequential(nn.BatchNorm1d(embedding_dim),
                        nn.ReLU(inplace=True),
                        nn.Dropout(p=0.5)).cuda()

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1).float()
        x=self.conv1(x)
        x=self.lin(x.squeeze(dim=1))
        x = self.fc1(x.squeeze(dim=1))
        x = nn.functional.normalize(x)
        return x

'''
Linear Weighted Addition
'''

class LinearWeightedAvg(nn.Module):
    '''
    Perform modality fusion through Linear weighted averaging
    :param feat_dim: input dimension of features previously extracted
    :param embedding_dim: output dimension of fetures embedding to give to fusion model
    '''
    def __init__(self):
        super(LinearWeightedAvg, self).__init__()
        self.weight1 = nn.Parameter(torch.rand(1, device='cuda')).requires_grad_()
        self.weight2 = nn.Parameter(torch.rand(1, device='cuda')).requires_grad_()
        self.weight3 = nn.Parameter(torch.rand(1, device='cuda')).requires_grad_()
    def forward(self, img_feat, audio_feat, txt_feat):
        return (self.weight1 * img_feat + self.weight2 * audio_feat + self.weight3 * txt_feat)


'''
Gated Multi-Modal Fusion
'''


class GatedFusion(nn.Module):
    '''
            Perform modality fusion through Gated Multi-modal fusion
            :param feat_dim: input dimension of features previously extracted
            :param embedding_dim: output dimension of fetures embedding to give to fusion model
    '''
    def __init__(self, embed_dim_in, mid_att_dim, emb_dim_out):
        super(GatedFusion, self).__init__()
        self.linear_img = nn.Sequential()
        self.linear_audio = nn.Sequential()
        self.linear_txt = nn.Sequential()
        self.final_transform = nn.Sequential()
        self.attention = nn.Sequential(
            Forward_Block(embed_dim_in * 2, mid_att_dim),
            nn.Linear(mid_att_dim, emb_dim_out)
        )

    def forward(self, img_input, audio_input, txt_input,):
        concat = torch.cat((img_input, audio_input, txt_input), dim=1)
        attention_out = torch.sigmoid(self.attention(concat))
        img_trans = torch.tanh(self.linear_img(img_input))
        audio_trans = torch.tanh(self.linear_audio(audio_input))
        txt_trans = torch.tanh(self.linear_txt(txt_input))

        # out = img_trans * attention_out + (1.0 - attention_out) * audio_trans  ########################
        out = img_trans * attention_out + audio_trans * attention_out + txt_trans * attention_out
        out = self.final_transform(out)

        return out, img_trans, audio_trans, txt_trans


class Forward_Block(nn.Module):

    def __init__(self, input_dim=128, output_dim=128, p_val=0.0):
        super(Forward_Block, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(p=p_val)
        )

    def forward(self, x):
        return self.block(x)


'''
Main Module
'''



class myModel(nn.Module):
    def __init__(self, fusion, dim_embed, image_feat_dim, audio_feat_dim,txt_feat_dim):
        super(myModel, self).__init__()

        self.audio_branch = EmbedBranch(audio_feat_dim, dim_embed)
        self.image_branch = EmbedBranch_Video(dim_embed)
        self.txt_branch = EmbedBranch(txt_feat_dim, dim_embed)
        if fusion == 'linear':
            self.fusion_layer = LinearWeightedAvg()
        # elif fusion == 'gated':
        #     self.fusion_layer = GatedFusion(image_feat_dim,audio_feat_dim,txt_feat_dim, dim_embed, 128, dim_embed)

        self.logits_layer = nn.Linear(dim_embed, 6)

        self.cuda()

    def forward(self, text_feat, audio_feat, img_feat, labels):
        audio = self.audio_branch(audio_feat)
        image = self.image_branch(img_feat)
        txt = self.txt_branch(text_feat)
        feats = self.fusion_layer(image, audio, txt)
        logits = self.logits_layer(feats)

        return (feats, logits)

    # def train_forward(self, image, audio, txt, labels):
    #
    #     comb, face_embeds, voice_embeds, txt_embeds = self(image, audio, txt)
    #     return comb, face_embeds, voice_embeds, txt_embeds


