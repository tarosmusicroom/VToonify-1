#@title **‰æ‘œ‚ÌƒAƒjƒ‰»**

with torch.no_grad():
    I = align_face(frame, landmarkpredictor)
    I = transform(I).unsqueeze(dim=0).to(device)
    s_w = pspencoder(I)
    s_w = vtoonify.zplus2wplus(s_w)
    s_w[:,:7] = exstyle[:,:7]
    # parsing network works best on 512x512 images, so we predict parsing maps on upsmapled frames
    # followed by downsampling the parsing maps
    x_p = F.interpolate(parsingpredictor(2*(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)))[0], 
                        scale_factor=0.5, recompute_scale_factor=False).detach()
    # we give parsing maps lower weight (1/16)
    inputs = torch.cat((x, x_p/16.), dim=1)
    # d_s has no effect when backbone is toonify
    y_tilde = vtoonify(inputs, s_w.repeat(inputs.size(0), 1, 1), d_s = 0.5)        
    y_tilde = torch.clamp(y_tilde, -1, 1)

# ‰æ‘œo—Í
from google.colab.patches import cv2_imshow
import cv2
clear_output()
save_image(y_tilde[0].cpu(), 'after.jpg')
img_befor = cv2.imread('befor.jpg')
tmp = cv2.imread('after.jpg')
img_after = cv2.resize(tmp, dsize=(W, H))
img = cv2.hconcat([img_befor, img_after])
cv2.imwrite('img.jpg', img)
cv2_imshow(img)