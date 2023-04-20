import os

def zip_files(current_loop_num):
    if os.path.exists(f'./results/generated_images{current_loop_num}'):
        os.chdir(f'./results/generated_images{current_loop_num}')
        os.system("pwd")
        os.system(f'tar -zcvf ../../images{current_loop_num}.tgz *.jpg')
        os.chdir('../..')
    else:
        raise

def rename_files(current_loop_num):
    """Rename all .jpg files in the current directory."""
    # Get all the files in the current directory
    files = os.listdir('./results/default')
    # Loop through the files
    file_num = 1
    for f in files:
        # Check if the file is a .jpg file
        if f.startswith('generated') and f.endswith('.jpg') and f.startswith('generated') and (not f.endswith('-ema.jpg')) and (not f.endswith('-mr.jpg')):
            print(f"renaming {f} to {str(file_num)}.jpg")
            if file_num == 1:
                os.mkdir(f'./results/generated_images{str(current_loop_num)}')
            # Rename the file
            os.system('mv ./results/default/' + f + ' ./results/generated_images' + str(current_loop_num) + '/' + str(file_num) + '.jpg')
            file_num += 1
        elif (f.endswith('-ema.jpg') or f.endswith('-mr.jpg') ) and f.startswith('generated'):
            os.system('rm ./results/default/' + f)
    print(file_num)
    if file_num == 1:
        print("no files been compressed!!")
        raise

def remove_old_files():
    if os.path.exists('./images.tgz'):
        print("exist tgz file")
        os.system('rm ./images.tgz')
    if os.path.exists('./results/generated_images'):
        print("exist folder")
        os.system('rm -r ./results/generated_images')

def main():
    os.system(f"pip install stylegan2_pytorch")
    steps_per_loop = 20000
    current_loop_num = 2
    save_every = 1000
    while(True):
        # train
        load_from = int((steps_per_loop / save_every) * current_loop_num)
        current_loop_num += 1
        steps_num = int(steps_per_loop * current_loop_num)
        print(f"[INFO] loop {current_loop_num}, load from ckpt {load_from}, num-train-steps = {steps_num}, save_every={save_every}")

        if load_from == 0:
            os.system(f"stylegan2_pytorch --data faces --image-size 64 --batch-size 16 --num-train-steps={steps_num} --save-every={save_every}")
        else:
            os.system(f"stylegan2_pytorch --data faces --image-size 64 --batch-size 16 --num-train-steps={steps_num} --load-from {load_from} --save-every={save_every}") 
        
        print(f"[INFO] finish training this loop")

        # inference
        print(f"[INFO] generating images...")
        os.system(f"stylegan2_pytorch --generate --num_generate=1000 --num_image_tiles=1")
        print(f"[INFO] images generated !!!")

        # rename files
        print(f"[INFO] move files and rename them")
        rename_files(current_loop_num)

        # zip files
        print(f"[INFO] zipping files...")
        zip_files(current_loop_num)       

if __name__ == '__main__':
    main()
    # remove_old_files()
    # rename_files()
    # zip_files()