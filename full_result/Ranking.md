# Ranking of 80 Models on SPASL_v1

- The source of the model size, and top-1 accuracy on ImageNet is PyTorch. Details can be found at [MODELS AND PRE-TRAINED WEIGHTS](https://pytorch.org/vision/stable/models.html).
- The SPASL score is on a scale of 0 to 100, higher the better.
- Full performance data of each model can be found in ["SPAPS_v1_full_test_results.xlsx"](./SPAPS_v1_full_test_results.xlsx).
- Please forgive the informal format of the model name in the table below. **Orz**

| **Model** | **Size(MB)** | **ImageNet-acc1 (%)** | **SPASL score** |
|:---------:|:------------:|:-----------------:|:---------------:|
| vit_l_16  |1164.3 |88.064 |   51.1559934|
| vit_h_14  |2416.6 |88.552 |   41.8053528|
| maxvit_t  |118.8  |83.7   |   36.2588818|
| vit_b_32  |336.6  |75.912 |   34.921327|
| vit_l_32  |1169.4 |76.972 |   32.76645|
| swin_b    |335.4  |83.582 |   30.0738808|
| swin_v2_b |336.4  |84.112 |   29.768583|
| swin_v2_s |190.7  |83.712 |   29.0371554|
| swin_s    |189.8  |83.196 |   28.3100056|
| wide_resnet101_2  |484.7  |82.51  |   27.1949532|
| resnet152 |230.5  |82.284 |   27.1148094|
| resnext101_32x8d  |339.7  |82.834 |   27.060763|
| regnet_y_128gf    |2461.6 |88.228 |   26.3183954|
| resnext101_64x4d  |319.3  |83.246 |   25.8807014|
| regnet_y_16gf |319.5  |86.012 |   25.7888462|
| regnet_x_32gf |412    |83.014 |   25.514496|
| resnet101 |170.5  |81.886 |   25.0336706|
| regnet_y_8gf  |150.7  |82.828 |   24.3982058|
| regnet_x_16gf |207.6  |82.716 |   23.7700464|
| efficientnet_b4   |74.5   |83.384 |   23.7474758|
| efficientnet_v2_s |82.7   |84.228 |   23.5968384|
| efficientnet_v2_m |208    |85.112 |   23.184389|
| efficientnet_b3   |47.2   |82.008 |   22.9569702|
| swin_v2_t |108.6  |82.072 |   22.4263754|
| regnet_y_3_2gf    |74.6   |81.982 |   21.7971252|
| regnet_x_8gf  |151.5  |81.682 |   21.430014|
| swin_t    |108.2  |81.474 |   20.9587422|
| vit_b_16  |331.4  |85.304 |   20.7766588|
| resnext50_32x4d   |95.8   |81.198 |   20.7560826|
| convnext_large    |754.5  |84.414 |   20.7093306|
| wide_resnet50_2   |263.1  |81.602 |   20.5663148|
| regnet_x_3_2gf    |58.8   |81.196 |   20.5138372|
| regnet_y_1_6gf    |43.2   |80.876 |   20.2675124|
| convnext_base |338.1  |84.062 |   19.6036142|
| efficientnet_v2_l |454.6  |85.808 |   18.9300426|
| resnet50  |97.8   |80.858 |   18.4893716|
| convnext_small    |191.7  |83.616 |   18.2892294|
| convnext_tiny |109.1  |82.52  |   16.7859438|
| regnet_y_800mf    |24.8   |78.828 |   16.4184248|
| efficientnet_b1   |30.1   |79.838 |   15.503322|
| regnet_y_32gf |554.1  |86.838 |   15.1238618|
| densenet161   |110.4  |77.138 |   14.1364134|
| regnet_x_1_6gf    |35.3   |79.668 |   13.9670866|
| densenet201   |77.4   |76.896 |   12.7025658|
| densenet169   |54.7   |75.6   |   12.4900674|
| regnet_y_400mf    |16.8   |75.804 |   12.2208778|
| shufflenet_v2_x2_0    |28.4   |76.23  |   11.9132326|
| regnet_x_800mf    |27.9   |77.522 |   11.6325338|
| efficientnet_b2   |35.2   |80.608 |   11.588342|
| inception_v3  |103.9  |77.294 |   11.3781584|
| mobilenet_v3_large    |21.1   |75.274 |   10.9512974|
| mnasnet1_3    |24.2   |76.506 |   10.0313396|
| regnet_x_400mf    |21.3   |74.864 |   9.9675958|
| densenet121   |30.8   |74.434 |   9.9662064|
| efficientnet_b7   |254.7  |84.122 |   9.5171922|
| googlenet |49.7   |69.778 |   9.4233272|
| efficientnet_b6   |165.4  |84.008 |   9.409512|
| shufflenet_v2_x1_5    |13.6   |72.996 |   8.379565|
| efficientnet_b0   |20.5   |77.692 |   8.305159|
| mobilenet_v3_small    |9.8    |67.668 |   7.4800826|
| efficientnet_b5   |116.9  |83.444 |   7.2284402|
| mobilenet_v2  |13.6   |72.154 |   6.147398|
| mnasnet0_75   |12.3   |71.18  |   5.7996346|
| resnet34  |83.3   |73.314 |   5.402992|
| vgg19_bn  |548.1  |74.218 |   4.1090984|
| vgg16_bn  |527.9  |73.36  |   3.6124434|
| resnet18  |44.7   |69.758 |   2.884633|
| mnasnet1_0    |16.9   |73.456 |   2.5979118|
| vgg13_bn  |507.6  |71.586 |   2.527274|
| vgg11_bn  |506.9  |70.37  |   2.1847704|
| vgg19 |548.1  |72.376 |   1.8737774|
| shufflenet_v2_x1_0    |8.8    |69.362 |   1.8604702|
| vgg16 |527.8  |71.592 |   1.8470332|
| vgg13 |507.5  |69.928 |   1.3376778|
| vgg11 |506.8  |69.02  |   1.0826156|
| mnasnet0_5    |8.6    |67.734 |   1.007643|
| shufflenet_v2_x0_5    |5.3    |60.552 |   0.5371038|
| alexnet   |233.1  |56.522 |   0.3473624|
| squeezenet1_0 |4.8    |58.092 |   0.2938566|
| squeezenet1_1 |4.7    |58.178 |   0.2598068|

