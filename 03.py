#@title **画像のクロップ**
pic = '01.jpg' #@param {type:"string"}
image_path = './data/'+pic

frame = cv2.imread(image_path)
frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

scale = 1
kernel_1d = np.array([[0.125],[0.375],[0.375],[0.125]])
# We detect the face in the image, and resize the image so that the eye distance is 64 pixels.
# Centered on the eyes, we crop the image to almost 400x400 (based on args.padding).
paras = get_video_crop_parameter(frame, landmarkpredictor, padding=[200,200,200,200])
if paras is not None:
    h,w,top,bottom,left,right,scale = paras
    H, W = int(bottom-top), int(right-left)
    # for HR image, we apply gaussian blur to it to avoid over-sharp stylization results
    if scale <= 0.75:
        frame = cv2.sepFilter2D(frame, -1, kernel_1d, kernel_1d)
    if scale <= 0.375:
        frame = cv2.sepFilter2D(frame, -1, kernel_1d, kernel_1d)
    frame = cv2.resize(frame, (w, h))[top:bottom, left:right]
    x = transform(frame).unsqueeze(dim=0).to(device)
else:
    print('no face detected!')

# 画像出力
save_image(x[0].cpu(),'befor.jpg')
visualize(x[0].cpu(), 60)