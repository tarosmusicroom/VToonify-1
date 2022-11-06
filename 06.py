#@title **ìÆâÊÇÃÉAÉjÉÅâª**
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
videoWriter = cv2.VideoWriter(os.path.join(OUT_DIR, 'result.mp4'), fourcc, video_cap.get(5), (4*W, 4*H))
batch_size = 4

with torch.no_grad():
    batch_frames = []
    for i in tqdm(range(num)):
        if i == 0:        
            I = align_face(frame, landmarkpredictor)
            I = transform(I).unsqueeze(dim=0).to(device)
            s_w = pspencoder(I)
            s_w = vtoonify.zplus2wplus(s_w)
            s_w[:,:7] = exstyle[:,:7]
        else:
            success, frame = video_cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if scale <= 0.75:
                frame = cv2.sepFilter2D(frame, -1, kernel_1d, kernel_1d)
            if scale <= 0.375:
                frame = cv2.sepFilter2D(frame, -1, kernel_1d, kernel_1d)
            frame = cv2.resize(frame, (w, h))[top:bottom, left:right]

        batch_frames += [transform(frame).unsqueeze(dim=0).to(device)]

        if len(batch_frames) == batch_size or (i+1) == num:
            x = torch.cat(batch_frames, dim=0)
            batch_frames = []
            # parsing network works best on 512x512 images, so we predict parsing maps on upsmapled frames
            # followed by downsampling the parsing maps
            x_p = F.interpolate(parsingpredictor(2*(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)))[0], 
                            scale_factor=0.5, recompute_scale_factor=False).detach()
            # we give parsing maps lower weight (1/16)
            inputs = torch.cat((x, x_p/16.), dim=1)
            # d_s has no effect when backbone is toonify
            y_tilde = vtoonify(inputs, s_w.repeat(inputs.size(0), 1, 1), d_s = 0.5)       
            y_tilde = torch.clamp(y_tilde, -1, 1)
            for k in range(y_tilde.size(0)):
                videoWriter.write(tensor2cv2(y_tilde[k].cpu()))
videoWriter.release()
video_cap.release()

# ìÆâÊèoóÕ
clear_output()
re_size = str(W)+'x'+str(H)
! ffmpeg -y -i ./output/result.mp4 -vcodec h264 -pix_fmt yuv420p -s $re_size -loglevel error after.mp4
! ffmpeg -y -i befor.mp4 -i after.mp4 -filter_complex "hstack" -loglevel error output.mp4
display_mp4('output.mp4')