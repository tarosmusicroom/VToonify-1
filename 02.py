#@title **ƒ‚ƒfƒ‹‚Ì‘I‘ð**

style_type = "arcane077" #@param ["cartoon026", "cartoon299", "arcane000", "arcane077", "pixar052", "caricature039", "caricature068", "comic000", "comic100"]



transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5,0.5,0.5]),
    ])

vtoonify = VToonify(backbone = 'dualstylegan')
vtoonify.load_state_dict(torch.load(os.path.join(MODEL_DIR, style_type+'_generator.pt'), map_location=lambda storage, loc: storage)['g_ema'])
vtoonify.to(device)

parsingpredictor = BiSeNet(n_classes=19)
parsingpredictor.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'faceparsing.pth'), map_location=lambda storage, loc: storage))
parsingpredictor.to(device).eval()

modelname = './checkpoint/shape_predictor_68_face_landmarks.dat'
if not os.path.exists(modelname):
    import wget, bz2
    wget.download('http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2', modelname+'.bz2')
    zipfile = bz2.BZ2File(modelname+'.bz2')
    data = zipfile.read()
    open(modelname, 'wb').write(data) 
landmarkpredictor = dlib.shape_predictor(modelname)

pspencoder = load_psp_standalone(os.path.join(MODEL_DIR, 'encoder.pt'), device)    

exstyles = np.load(os.path.join(MODEL_DIR, style_type[:-3]+'_exstyle_code.npy'), allow_pickle='TRUE').item()
stylename = list(exstyles.keys())[int(style_type[-3:])]
exstyle = torch.tensor(exstyles[stylename]).to(device)
with torch.no_grad():  
    exstyle = vtoonify.zplus2wplus(exstyle)
    
print('Model successfully loaded!')