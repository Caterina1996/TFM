
#Asigna la emoción que mayor probabilidad tiene en la
#distribución de probabilidades de esa imagen

def get_emo(emo_info):
	idx=emo_info.index(max(emo_info))
	return idx

#Leemos el fichero distribution txt
#Este fichero contiene la distribución de probabilidades para cada imagen para cada una de las 6 emociones
#Cada columna se corresponde con Surprise, Fear, Disgust, Happiness, Sadness and Anger, respectivamente.
f = open("distribution.txt", "r")
labels = f.readlines()
f.close()

#Leemos el fichero de partition_label.txt
#Contiene la distribución train/test que hicieron ellos del dataset
# each row: <image_name> <partition label>
# label 0 for train
# label 1 for test
f = open("partition_label.txt", "r")
partition = f.readlines()
f.close()

split_keys=["train","test"]
emo_labels=["Surprise","Fear","Disgust","Happiness","Sadness","Anger"]

for p,l in zip(partition,labels):
	
	tr_or_val = p.split()
	#nombre de la imagen:
	file=tr_or_val.pop(0)
	print("file ",file)

	#Asignamos la etiqueta correspondiente train or test
	train_valid=split_keys[int(tr_or_val[0])]

	emo = l.split()
	emo.pop(0)

	#Asignamos la etiqueta de emoción correspondiente:
	emotion=emo_labels[get_emo(emo)]

	#Generamos los comandos para guardar la imagen en la carpeta correspondiente:
	txt = "cp ./aligned/{}_aligned.jpg  ./dataset_ordenat/{}/{}".format(file[:-4],train_valid,emotion)
	print(txt)

#Instrucciones:
#1)Ejecutamos este archivo para generar un fichero .bat
#Ejecutar como python ./tidyData.py > ordenardataset.bat
#2)Ejecutar el fichero ordenardataset.bat