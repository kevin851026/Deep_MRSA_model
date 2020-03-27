import json
import matplotlib.pyplot as plt
with open('loss.json' , 'r') as reader:
	jf = json.loads(reader.read())
	v_loss=jf['v_loss']
	loss=jf['loss']
	epoch=jf['epoch']
	plt.clf()
	plt.plot(epoch,loss,lw=3,label="training data")
	plt.plot(epoch,v_loss,"r",lw=3,label="validation data")
	plt.xlabel("epoch")
	plt.ylabel("loss")
	plt.ylim((1,3))
	plt.yticks([1,1.5,2,2.5,3])
	plt.legend(loc="best")
	plt.savefig('pic.jpg')
	
