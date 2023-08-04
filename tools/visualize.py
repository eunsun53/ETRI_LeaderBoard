import matplotlib.pyplot as plt 

class ResultVisualize():

    def __init__(self, df):
        self.df = df 
        self.n = len(df) #epoch
        self.epoch_list = list(range(1, self.n+1)) #epoch list 생성 
        
    def drawGraphOnegraph(self, save_path):        
        #plot 그리기 
        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('loss', color = color)
        ax1.plot(self.epoch_list, self.df['loss'], color=color, label='train') #실선
        # ax1.plot(epoch_list, val_loss, linestyle='--', color=color, label='val') #점선 
        ax1.tick_params(axis='y', labelcolor=color)

        # ax2 = ax1.twinx() 

        # color = 'tab:blue'
        # ax2.set_ylabel('Accuracy', color=color)
        # ax2.plot(self.epoch_list, self.df['acc'], color=color, label='train')
        # # ax2.plot(epoch_list, val_accuracy, linestyle='--', color=color, label='val')
        # ax2.tick_params(axis='y', labelcolor=color)

        # 그래프에 legend 추가
        lines1, labels1 = ax1.get_legend_handles_labels()
        # lines2, labels2 = ax2.get_legend_handles_labels()
        # lines = lines1 + lines2
        # labels = labels1 + labels2
        ax1.legend(lines1, labels1, loc='right') #'lower right', 'lower left', 'upper right', 'center', 'right', 'left', ...

        fig.tight_layout()
        plt.savefig(save_path) #이미지 저장 
        print(f"========={save_path} 그래프 저장 완료!!==========")
        # plt.show()

    def drawGraphEachgraph(self, save_path):
        plt.figure(figsize = (8, 3))

        plt.subplot(121) #서브 플롯 그리기 (1, 2)로 나눈 서브플롯 중 1번째 
        plt.plot(self.df['acc'])
        plt.title("acc") #그래프 제목 
        plt.xlabel('Epoch') #x축 이름  
        plt.ylabel('Acc') #y축 이름 
        plt.legend(['train'], loc='upper right') #그래프 이름 

        plt.subplot(122)
        plt.plot(self.df['loss'])
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(['train'], loc='lower right')

        plt.tight_layout()
        plt.savefig(save_path) #이미지 저장 
        print(f"========={save_path} 그래프 저장 완료!!==========")
        # plt.show()
