
IMAGE_W_LABEL_TXT = '/home/ubuntu/Workspace/blur_recog/zhenai_aligned_0_blurred_processed.txt'

txt_path = IMAGE_W_LABEL_TXT



with open(txt_path, 'r') as file:
  fh = file.readlines()
  imgs = []
  for line in fh:
    print(line)
    line = line.rstrip()
    file_name, label = line.split()
    imgs.append((file_base_dir + '/' + file_name, int(label)))


for i in range(5):
  print(fh[i])