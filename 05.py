#@title **動画のクロップ**

video = '03.mp4' #@param {type:"string"}
video_path = './data/'+video
video_cap = cv2.VideoCapture(video_path)
num = int(video_cap.get(7))

success, frame = video_cap.read()
if success == False:
    assert('load video frames error')
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


scale = 1
kernel_1d = np.array([[0.125],[0.375],[0.375],[0.125]])
# We proprocess the video by detecting the face in the first frame, 
# and resizing the frame so that the eye distance is 64 pixels.
# Centered on the eyes, we crop the first frame to almost 400x400 (based on args.padding).
# All other frames use the same resizing and cropping parameters as the first frame.
paras = get_video_crop_parameter(frame, landmarkpredictor, padding=[200,200,200,200])
if paras is None:
    print('no face detected!')
else:
    h,w,top,bottom,left,right,scale = paras
    H, W = int(bottom-top), int(right-left)
# for HR video, we apply gaussian blur to the frames to avoid flickers caused by bilinear downsampling
# this can also prevent over-sharp stylization results. 
if scale <= 0.75:
    frame = cv2.sepFilter2D(frame, -1, kernel_1d, kernel_1d)
if scale <= 0.375:
    frame = cv2.sepFilter2D(frame, -1, kernel_1d, kernel_1d)
    
frame = cv2.resize(frame, (w, h))[top:bottom, left:right]


# 偶数補正
def even(x):
  if x % 2 != 0:
    x +=1
  return x
w = even(w)
h = even(h)

# ビデオクロップ
! ffmpeg -y -i $video_path -vf scale=$w:$h -loglevel error tmp.mp4
crop_size = 'crop='+str(W)+':'+str(H)+':'+str(left)+':'+str(top)
! ffmpeg -y -i tmp.mp4 -filter:v $crop_size -async 1 -loglevel error befor.mp4
display_mp4('befor.mp4') 
