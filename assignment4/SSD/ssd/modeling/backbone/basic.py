import torch
import torch.nn as nn


class BasicModel(torch.nn.Module):
    """
    This is a basic backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 38, 38),
     shape(-1, output_channels[1], 19, 19),
     shape(-1, output_channels[2], 10, 10),
     shape(-1, output_channels[3], 5, 5),
     shape(-1, output_channels[3], 3, 3),
     shape(-1, output_channels[4], 1, 1)]
     where "output_channels" is the same as cfg.BACKBONE.OUT_CHANNELS
    """
    def __init__(self, cfg):
        super().__init__()
        image_size = cfg.INPUT.IMAGE_SIZE # 300
        output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS # (512, 1024, 512, 256, 256, 256)
        self.output_channels = output_channels
        image_channels = cfg.MODEL.BACKBONE.INPUT_CHANNELS # 3 
        self.output_feature_size = cfg.MODEL.PRIORS.FEATURE_MAPS #[38, 19, 10, 5, 3, 1]
        self.num_filters = [
                    32, 64, 64, output_channels[0], 
                    64, output_channels[1], 
                    128, output_channels[2],
                    128, output_channels[3],
                    128, output_channels[4], 
                    128, output_channels[5]]

        # self.num_filters = [num_filter * 2 for num_filter in self.num_filters]        
        # self.output_feature_size = [output_feature_size[0] * 2  for output_feature_size in self.output_feature_size]        
        print(self.num_filters)
        # Define the convolutional layers
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=self.num_filters[0],
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.MaxPool2d((2,2), 2),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.num_filters[0],
                out_channels=self.num_filters[1],
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.MaxPool2d((2,2), 2),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.num_filters[1],
                out_channels=self.num_filters[2],
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.num_filters[1],
                out_channels=self.num_filters[2],
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.num_filters[1],
                out_channels=self.num_filters[2],
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.num_filters[1],
                out_channels=self.num_filters[2],
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),

                                                
            #first output convolutional
            nn.Conv2d(
                in_channels=self.num_filters[2],
                out_channels=self.num_filters[3],
                kernel_size=3,
                stride=2,
                padding=1
            ),


            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.num_filters[3],
                out_channels=self.num_filters[4],
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            #second output convolutional
            nn.Conv2d(
                in_channels=self.num_filters[4],
                out_channels=self.num_filters[5],
                kernel_size=3,
                stride=2,
                padding=1
            ),


            nn.Dropout(p=0.15),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.num_filters[5],
                out_channels=self.num_filters[6],
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            #third output convolutional
            nn.Conv2d(
                in_channels=self.num_filters[6],
                out_channels=self.num_filters[7],
                kernel_size=3,
                stride=2,
                padding=1
            ),


            nn.Dropout(p=0.15),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.num_filters[7],
                out_channels=self.num_filters[8],
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            #fourth output convolutional
            nn.Conv2d(
                in_channels=self.num_filters[8],
                out_channels=self.num_filters[9],
                kernel_size=3,
                stride=2,
                padding=1
            ),


            nn.Dropout(p=0.15),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.num_filters[9],
                out_channels=self.num_filters[10],
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            #fifth output convolutional
            nn.Conv2d(
                in_channels=self.num_filters[10],
                out_channels=self.num_filters[11],
                kernel_size=3,
                stride=2,
                padding=1
            ),


            nn.Dropout(p=0.15),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.num_filters[11],
                out_channels=self.num_filters[12],
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),

            #sixth output convolutional
            nn.Conv2d(
                in_channels=self.num_filters[12],
                out_channels=self.num_filters[13],
                kernel_size=3,
                stride=1,
                padding=0
            ),
        )


    def forward(self, x):
        """
        The forward functiom should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[3], 3, 3),
            shape(-1, output_channels[4], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """

        out_features = []
        for i in range(9+6):
            x = self.feature_extractor[i](x)
        out_features.append(x)

        for i in range(9+6, 13+6):
            x = self.feature_extractor[i](x)
        out_features.append(x)

        for i in range(13+6, 18+6):
            x = self.feature_extractor[i](x)
        out_features.append(x)

        for i in range(18+6, 23+6):
            x = self.feature_extractor[i](x)
        out_features.append(x)  

        for i in range(23+6, 28+6):
            x = self.feature_extractor[i](x)
        out_features.append(x)

        for i in range(28+6, 33+6):
            x = self.feature_extractor[i](x)
        out_features.append(x)
        for idx, feature in enumerate(out_features):
            expected_shape = (self.output_channels[idx], self.output_feature_size[idx], self.output_feature_size[idx])
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        return tuple(out_features)

