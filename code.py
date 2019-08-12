
import numpy as np
from lightfm import LightFM
from sklearn import preprocessing
from lightfm.data import Dataset
from lightfm.evaluation import auc_score
from scipy.sparse import csr_matrix 
from scipy.sparse import coo_matrix 
from sklearn.model_selection import train_test_split
#from keras.utils.np_utils import to_categorical   
#from sklearn.preprocessing import OneHotEncoder


##THIS FUNCTION GENERATES FAKE DATA: USERID, ITEM ID, RATING (1 TO 10)
def create_data(n):
    users=np.random.choice(np.arange(n),(n,1))
    items=np.random.choice(['p1','p2','p4','p12','p13','p41','p10','p60','p42',\
                            'p11','p21','p456','p342','p513','p941','p130','p160','p942','p342','p5k13','p9v41','p1630','p1630','p94h32','p1b0','p6a0','pt42','pe11','p2x1','pd456','p3de42','p5hg13','p9r41','ps130','p1x60','pb942','pa342','p5ek13','p9vv41','p163d0','p16d30','p943e2'],(n,1))
    rating=np.random.randint(1,high=10,size=(n,1),dtype='int')
    return np.column_stack([users,items,rating])
    
data=create_data(1000)


users_encod=preprocessing.LabelEncoder()
users_recod=users_encod.fit_transform(data[:,0])

items_encod=preprocessing.LabelEncoder()
items_recod=items_encod.fit_transform(data[:,1])

genr_encod=preprocessing.LabelEncoder()
genr_recod=genr_encod.fit_transform(data[:,3])

age_encod=preprocessing.LabelEncoder()
age_recod=age_encod.fit_transform(data[:,4])


train=np.column_stack((users_recod,items_recod,data[:,2].astype(int)))

####TRAIN SET PARAMS

n_users=np.unique(train[:,0]).shape[0]
n_items=np.unique(train[:,1]).shape[0]

##second position are ratings, 0 are user ids, 1 are item_ids

user_interact=coo_matrix((train[:,2],(train[:,0],train[:,1])),shape=(n_users,n_items))


##TRANING LIGHTFM

model=LightFM(no_components=20,learning_rate=0.1,loss='warp')
model.fit(interactions=user_interact,epochs=20,verbose=True)


###GET PREDICTIONS

def predict_mod(modelfm,user_list,items_list,top=3):
    
    rec=[]
   # username=[]
    
    for i in user_list:
        score=modelfm.predict([i],item_ids=items_list)
        #score=modelfm.predict([i],item_ids=items_list,user_features=userfeats)
        rec.append(items_encod.classes_[np.argsort(-score)[:top]])

    return(rec)

predict_mod(model,np.arange(n_users),np.arange(items_encod.classes_.shape[0]))







    
