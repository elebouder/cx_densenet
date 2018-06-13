import os


test_src = '/home/elebouder/Data/landsat/training_clips/Test/'
train_src = '/home/elebouder/Data/landsat/training_clips/Train/'
train_dst = '/home/elebouder/Data/landsat/train_llbl.txt'
test_dst = '/home/elebouder/Data/landsat/test_llbl.txt'




def load_data(src):
    l = []
    for fname in os.listdir(src):
        if fname.endswith("tif"):
            #uncomment following line if you want to use png instead of tif
            png_name = fname.split(".")[0] + ".png"
            label = int(fname.split(".")[0].split("_")[-1])
            l.append([png_name, label])
    return l


def generate_list():
	if os.path.exists(test_dst):
		os.remove(test_dst)
	if os.path.exists(train_dst):
		os.remove(train_dst)
	train_dict = load_data(train_src)
	f = open(train_dst, 'w')
	for elem in train_dict:
		f.write(elem[0] + " " + str(elem[1]))
		f.write("\n")
	
	f.close()

        test_dict = load_data(test_src)
        f = open(test_dst, 'w')
        for elem in test_dict:
                f.write(elem[0] + " " + str(elem[1]))
                f.write("\n")

        f.close()



generate_list()
