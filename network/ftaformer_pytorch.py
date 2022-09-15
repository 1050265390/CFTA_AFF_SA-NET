# FTANet
import torch
import torch.nn as nn
import torch.nn.functional as F

from network.attention_layer import CombineLayer, PositionalEncoding


class Local_Att(nn.Module):
    def __init__(self, in_channels, out_channels, ratio, limitation=4):
        super(Local_Att, self).__init__()
        inter_channels = max(int(out_channels // ratio), limitation)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        out = self.conv1(x)

        return out


class Global_Att(nn.Module):
    def __init__(self, in_channels, out_channels, ratio, limitation=4):
        super(Global_Att, self).__init__()
        inter_channels = max(int(out_channels // ratio), limitation)

        self.conv2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = self.conv2(x)

        return x


class Fusion(nn.Module):
    def __init__(self, in_channels, out_channels, ratio, limitation, mode=0):
        super(Fusion, self).__init__()

        self.mode = mode
        self.local_att = Local_Att(in_channels, out_channels, ratio, limitation)
        self.global_att = Global_Att(in_channels, out_channels, ratio, limitation)

    def forward(self, t, f):
        xa = t + f
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = nn.Softmax(dim=1)(xlg)

        xi = t * wei + f * (1 - wei)

        # x_t_l = self.local_att(t)
        # x_t_g = self.global_att(t)
        # xlg_t = x_t_l + x_t_g
        # wei = nn.Softmax(dim=1)(xlg_t)
        #
        # x_f_l = self.local_att_f(f)
        # x_f_g = self.global_att_f(f)
        # xlg_f = x_f_l + x_f_g
        # wei_f = nn.Softmax(dim=1)(xlg_f)
        # xi = t * wei + f * wei_f

        # AFF
        if self.mode == 0:
            return xi
        # iAFF
        else:
            xl2 = self.local_att(xi)
            xg2 = self.global_att(xi)
            xlg2 = xl2 + xg2
            wei2 = nn.Softmax(dim=1)(xlg2)

            xi = t * wei2 + f * (1 - wei2)
            return xi


class Fusion_All(nn.Module):
    def __init__(self, in_channels, out_channels, ratio, limitation):
        super(Fusion_All, self).__init__()

        self.fuse1 = Fusion(in_channels, out_channels, ratio, limitation)
        self.fuse2 = Fusion(in_channels, out_channels, ratio, limitation)
        self.fuse3 = Fusion(in_channels, out_channels, ratio, limitation)
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (5, 5), padding=2),
            nn.BatchNorm2d(out_channels),
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # x1 = self.conv3(x[0])
        # x2 = self.conv4(x[0])
        # x_x = self.fuse1(x1, x2)
        # x_x = self.bn1(x_x)
        # x_t = self.fuse2(x[0], x[1])
        # x_t = self.bn2(x_t)
        # x_f = self.fuse3(x[0], x[2])
        # x_f = self.bn3(x_f)
        # out = x_x * x_t * x_f

        out = self.fuse3(x[0], x[1])
        # out = x[0] + x[3]
        # out = self.fuse3(x[1], x[2])
        # out = self.bn3(out)

        return out


class FTA_Module(nn.Module):
    def __init__(self, shape, kt, kf):
        super(FTA_Module, self).__init__()
        self.bn = nn.BatchNorm2d(shape[2])
        self.r_cn = nn.Sequential(
            nn.Conv2d(shape[2], shape[3], (1, 1)),
            nn.ReLU()
        )

        self.t_dim = shape[3]
        self.t_token = shape[1]
        self.attn_dim = self.t_dim
        self.t_in = nn.Linear(self.t_dim, self.attn_dim)
        self.t_posenc = PositionalEncoding(self.attn_dim, n_position=self.t_token)
        self.t_dropout = nn.Dropout(p=0.2)
        self.t_norm = nn.LayerNorm(self.attn_dim, eps=1e-6)
        self.t_attn = nn.ModuleList([
            CombineLayer(self.attn_dim, self.attn_dim * 2, 8,
                         self.attn_dim // 8, self.attn_dim // 8, dropout=0.2)
            for _ in range(2)]
        )
        self.f_dim = shape[3]
        self.f_token = shape[0]
        self.attn_dim = self.f_dim
        self.f_in = nn.Linear(self.f_dim, self.attn_dim)
        self.f_posenc = PositionalEncoding(self.attn_dim, n_position=self.f_token)
        self.f_dropout = nn.Dropout(p=0.2)
        self.f_norm = nn.LayerNorm(self.attn_dim, eps=1e-6)
        self.f_attn = nn.ModuleList([
            CombineLayer(self.attn_dim, self.attn_dim * 2, 8,
                         self.attn_dim // 8, self.attn_dim // 8, dropout=0.2)
            for _ in range(2)]
        )

        self.ta_cn1 = nn.Sequential(
            nn.Conv1d(shape[2], shape[3], kt, padding=(kt - 1) // 2),
            nn.SELU()
        )
        self.ta_cn2 = nn.Sequential(
            nn.Conv1d(shape[3], shape[3], kt, padding=(kt - 1) // 2),
            nn.SELU()
        )
        self.ta_cn3 = nn.Sequential(
            nn.Conv2d(shape[2], shape[3], 3, padding=1),
            nn.SELU()
        )
        self.ta_cn4 = nn.Sequential(
            nn.Conv2d(shape[3], shape[3], 5, padding=2),
            nn.SELU()
        )

        self.fa_cn1 = nn.Sequential(
            nn.Conv1d(shape[2], shape[3], kf, padding=(kf - 1) // 2),
            nn.SELU()
        )
        self.fa_cn2 = nn.Sequential(
            nn.Conv1d(shape[3], shape[3], kf, padding=(kf - 1) // 2),
            nn.SELU()
        )
        self.fa_cn3 = nn.Sequential(
            nn.Conv2d(shape[2], shape[3], 3, padding=1),
            nn.SELU()
        )
        self.fa_cn4 = nn.Sequential(
            nn.Conv2d(shape[3], shape[3], 5, padding=2),
            nn.SELU()
        )

    def t_decoder(self, t_feature):
        t_h = self.t_dropout(self.t_posenc(t_feature))
        t_h = self.t_norm(t_h)
        for t_layer in self.t_attn:
            t_h, t_weight = t_layer(t_h, slf_attn_mask=None)
        return t_h

    def f_decoder(self, f_feature):
        f_h = self.f_dropout(self.f_posenc(f_feature))
        f_h = self.f_norm(f_h)
        for f_layer in self.f_attn:
            f_h, f_weight = f_layer(f_h, slf_attn_mask=None)
        return f_h

    def forward(self, x):

        x = self.bn(x)
        x_r = self.r_cn(x)
        #
        # # a_t = torch.mean(x_r, dim=-2)
        # # a_t = a_t.permute(0, 2, 1)
        # #
        # # a_t = self.t_decoder(a_t)
        # # a_t = nn.Softmax(dim=-2)(a_t)
        # # a_t = a_t.unsqueeze(dim=1)
        # # x_t = self.ta_cn3(x_r)
        # # x_t = self.ta_cn4(x_t)
        # # a_t = a_t.permute(0, 3, 1, 2)
        # # x_t = x_t * a_t
        # #
        # # a_f = torch.mean(x_r, dim=-1)
        # # a_f = a_f.permute(0, 2, 1)
        # # a_f = self.f_decoder(a_f)
        # # a_f = nn.Softmax(dim=-2)(a_f)
        # # a_f = a_f.unsqueeze(dim=2)
        # #
        # # x_f = self.fa_cn3(x_r)
        # # x_f = self.fa_cn4(x_f)
        # #
        # # a_f = a_f.permute(0, 3, 1, 2)
        # # x_f = x_f * a_f
        #
        #
        a_t = torch.mean(x, dim=-2)
        a_t = self.ta_cn1(a_t)
        a_t = self.ta_cn2(a_t)
        a_t = nn.Softmax(dim=-1)(a_t)
        a_t = a_t.unsqueeze(dim=-2)
        x_t = self.ta_cn3(x)
        x_t = self.ta_cn4(x_t)
        x_t = x_t * a_t

        a_f = torch.mean(x, dim=-1)
        a_f = self.fa_cn1(a_f)
        a_f = self.fa_cn2(a_f)
        a_f = nn.Softmax(dim=-1)(a_f)
        a_f = a_f.unsqueeze(dim=-1)
        x_f = self.fa_cn3(x)
        x_f = self.fa_cn4(x_f)
        x_f = x_f * a_f

        # x = self.fa_cn3(x)
        # x = self.fa_cn4(x)

        return x_r,(x_t + x_f)


class FTAFormer(nn.Module):
    def __init__(self, freq_bin=320, time_segment=128):
        super(FTAFormer, self).__init__()
        self.bn_layer = nn.BatchNorm2d(3)
        # bm
        self.bm_layer = nn.Sequential(
            nn.Conv2d(32, 16, (4, 1), stride=(4, 1)),
            nn.SELU(),
            nn.Conv2d(16, 16, (4, 1), stride=(4, 1)),
            nn.SELU(),
            nn.Conv2d(16, 16, (4, 1), stride=(4, 1)),
            nn.SELU(),
            nn.Conv2d(16, 1, (5, 1), stride=(5, 1)),
            nn.SELU()
        )

        # fta_module
        self.fta_1 = FTA_Module((freq_bin, time_segment, 3, 32), 3, 3)
        self.fta_2 = FTA_Module((freq_bin // 2, time_segment // 2, 32, 64), 3, 3)
        self.fta_3 = FTA_Module((freq_bin // 4, time_segment // 4, 64, 128), 3, 3)
        self.fta_4 = FTA_Module((freq_bin // 4, time_segment // 4, 128, 128), 3, 3)
        self.fta_5 = FTA_Module((freq_bin // 2, time_segment // 2, 128, 64), 3, 3)
        self.fta_6 = FTA_Module((freq_bin, time_segment, 64, 32), 3, 3)
        self.fta_7 = nn.Sequential(
            nn.Conv2d(32, 1, 3, padding=1),
            nn.SELU(),
            nn.Conv2d(1, 1, 5, padding=2),
            nn.SELU()
        )

        self.sf_1 = Fusion_All(32, 32, 4, 4)
        self.sf_2 = Fusion_All(64, 64, 4, 4)
        self.sf_3 = Fusion_All(128, 128, 4, 4)
        self.sf_4 = Fusion_All(128, 128, 4, 4)
        self.sf_5 = Fusion_All(64, 64, 4, 4)
        self.sf_6 = Fusion_All(32, 32, 4, 4)

        # maxpool
        self.mp_1 = nn.MaxPool2d((2, 2), (2, 2))
        self.mp_2 = nn.MaxPool2d((2, 2), (2, 2))
        self.up_1 = nn.Upsample(scale_factor=2)
        self.up_2 = nn.Upsample(scale_factor=2)

        self.dim = 320
        self.token = 128
        self.attn_dim = self.dim
        self.feature_in = nn.Linear(self.dim, self.attn_dim)
        self.feature_posenc = PositionalEncoding(self.attn_dim, n_position=self.token)
        self.feature_dropout = nn.Dropout(p=0.2)
        self.feature_norm = nn.LayerNorm(self.attn_dim, eps=1e-6)
        self.feature_attn = nn.ModuleList([
            CombineLayer(self.attn_dim, self.attn_dim * 2, 8,
                         self.attn_dim // 8, self.attn_dim // 8, dropout=0.2)
            for _ in range(2)]
        )
        self.final_layer = nn.Sequential(
            nn.Linear(self.attn_dim, 512),
            nn.Dropout(p=0.2),
            nn.SELU(),
            nn.Linear(512, 321),
            nn.Dropout(p=0.2),
            nn.SELU(),

        )
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         torch.nn.init.xavier_normal_(m.weight.data)
        #         # nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        #         if m.bias is not None:
        #             m.bias.data.zero_()
        #             # nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        #         # nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.Linear):
        #         torch.nn.init.normal_(m.weight.data, 0, 0.01)
        #         # m.weight.data.normal_(0,0.01)
        #         if m.bias is not None:
        #             m.bias.data.zero_()

    def feature_decoder(self, feature):
        feature_h = self.feature_dropout(self.feature_posenc(feature))
        feature_h = self.feature_norm(feature_h)
        for feature_layer in self.feature_attn:
            feature_h, feature_weight = feature_layer(feature_h, slf_attn_mask=None)
        return feature_h

    def forward(self, x):
        x = self.bn_layer(x)

        x1 = self.fta_1(x)
        x1 = self.sf_1(x1)
        x1 = self.mp_1(x1)

        x2 = self.fta_2(x1)
        x2 = self.sf_2(x2)
        x2 = self.mp_2(x2)

        x3 = self.fta_3(x2)
        x3 = self.sf_3(x3)

        x4 = self.fta_4(x3)
        x4 = self.sf_4(x4)

        x4 = self.up_1(x4)

        x5 = self.fta_5(x4)
        x5 = self.sf_5(x5)
        x5 = self.up_2(x5)

        x6 = self.fta_6(x5)
        x6 = self.sf_6(x6)

        x = self.fta_7(x6)

        bm = self.bm_layer(x6)
        x = x.squeeze(dim=1).permute(0, 2, 1)
        # # # #
        x = self.feature_decoder(x)
        # x = self.final_layer(x)
        x = x.permute(0, 2, 1).unsqueeze(dim=1)
        output_pre = torch.cat([bm, x], dim=2)
        output = nn.Softmax(dim=-2)(output_pre)
        # output2_1 = output[:, :, 0, :].unsqueeze(dim=2)
        # output2_2 = torch.sum(output[:, :, 1:321, :], dim=2).unsqueeze(dim=2)
        #
        # output2 = torch.cat([output2_1, output2_2], dim=2)
        # output2 = nn.Softmax(dim=-2)(output2)
        # print(output.shape, )
        return output, output

# from torchkeras import summary
#
# net = FTAFormer()
# summary(net, input_shape=(3, 320, 128))
