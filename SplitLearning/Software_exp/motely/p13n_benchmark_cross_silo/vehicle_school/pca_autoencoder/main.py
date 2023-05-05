import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
class AutoEncoder():
    def __init__(self, n, D):
        self.n=n
        self.D=D
        self.target=None
        self.pca_obj=None
    def train(self,data: np.ndarray):
        self.get_pca_components(data)
        pca=PCA(n_components=self.target)
        pca.fit_transform(data)
        self.pca_obj=pca
    def mse(x1:np.ndarray, x2:np.ndarray):
        if x1.shape!=x2.shape:
            print("Sizes don't match while calculating reconstruction loss")
            return None 
        else:
            M,N=x1.shape
            mse=(np.sum((x1-x2)**2))/(M*N)
            return mse
    def avg_recons_error(self,data:np.ndarray):
        embedding=self.pca_obj.transform(data)
        recons_data=self.pca_obj.inverse_transform(embedding)
        recons_loss=AutoEncoder.mse(data,recons_data)
        return recons_loss
    def recons_loss(self, test:np.ndarray):
        embedding=self.pca_obj.transform(test)
        recons_test=self.pca_obj.inverse_transform(embedding)
        recons_loss=AutoEncoder.mse(test,recons_test)
        return recons_loss
    def get_pca_components(self,data):
        n,D=data.shape
        if n<D:
            aaT=data@data.T
        else:
            aaT=data.T@data
        print("Matrix multiplication done")
        eigv,eigvec=np.linalg.eigh(aaT)
        print("Eigen values done")
        for i in range(eigv.shape[0]):
            if eigv[i]<0.0:
                eigv[i]=0
        eigv=eigv[::-1]
        sigma=np.sqrt(eigv)
        sigma_sum=np.sum(sigma**2)
        print("Sigma done")
        energy=0
        for i in range(sigma.shape[0]):  
            energy=np.sum(sigma[:i]**2)/sigma_sum
            if energy>=0.9:
                self.target=i 
                break
            else:
                continue

        
                
    

# def main():
#     (x_train,y_train),(x_test,y_test)=load_data()
#     split=train_test_split(x_train,train_size=0.01)
#     trainD=split[0]
#     testD=split[1]
#     testD=testD.reshape((testD.shape[0],784))
#     trainD=trainD.reshape((trainD.shape[0],784))
#     ae=AutoEncoder(trainD.shape[0],trainD.shape[1])
#     ae.train(trainD)
#     print(ae.avg_recons_error(trainD))
#     # print("Recons mnist: ",ae.recons_loss(trainD))
#     # (x_train,y_train),(x_test,y_test)=fashion_mnist.load_data()
#     # split2=train_test_split(x_train,train_size=0.8)
#     # trainD2=split[0]
#     # testD2=split[1]
#     # trainD2=trainD2.reshape((trainD.shape[0],784))
#     # testD2=testD2.reshape((testD.shape[0],784))
#     # print("Fmnist recons loss: ", ae.recons_loss(testD2))


# if __name__=="__main__":
#     main()
            
