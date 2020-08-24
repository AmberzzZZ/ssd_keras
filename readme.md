## SSD back
    SSD用vgg16做backbone，
    * 前四层conv+pooling和vgg16一样，conv4输出38x38x512
    * 第五个conv之后的pooling由2x2 s2变成 3x3 s1，输出19x19x1024
    * 接下来的尾巴是新增的
        ** conv6是3x3x1024的空洞卷积，atrous_rate=(6,6), 输出19x19x1024
        ** conv7是1x1x1024的conv，输出19x19x1024
        ** conv8是1x1x256和3x3x512 s2的conv，输出10x10x512
        ** conv9都是1x1x128和3x3x256 s2的conv，输出5x5x256
        ** conv10是1x1x128和3x3x256 s1 p0的conv，输出3x3x256
        ** 输入尺寸为300x300的条件下，conv11是1x1x128和3x3x256 s1 p0的conv，输出1x1x256
        ** 若输入尺寸大于300（512x512），conv11是GlobalAveragePooling，输出256
    
## 多尺度输出: 
    conv4 conv7 conv8 conv9 conv10 conv11都接上检测头作为输出
    n_boxes = [4,6,6,6,6,6]
    conv4: 38x38xn_boxesx(4+c)
    conv7: 19x19xn_boxesx(4+c)
    conv8: 10x10xn_boxesx(4+c)
    conv9: 5x5xn_boxesx(4+c)
    conv10: 3x3xn_boxesx(4+c)
    conv11: 1x1xn_boxesx(4+c)
    
    不同层针对不同的scale，scale的平方是面积，同一层相同面积不同ratio

## l2 norm on conv4:
    作者在parseNet中发现，网络conv4的特征scale和其他层不太一样，
    因此做了l2 norm，然后scale到一定大小，可以看作是一种instasnce norm

## loss



