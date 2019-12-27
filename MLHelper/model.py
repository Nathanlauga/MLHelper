import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_curve, auc,  precision_score, recall_score
from IPython.display import Markdown, display


class ClassifierAnalysis:
    """
    Class to analyse a classifier model that just has been trained.
    
    """
    
    def __init__(self, clf):
        self.clf = clf
        
    def compute_prediction(self, X):
        proba, preds = self.predict(X)
        
        self.y_proba = proba
        self.y_preds = preds
    
    def predict(self, X, threshold=0.5):
        proba = self.clf.predict_proba(X)[:, 1]
        return proba, (proba >= threshold).astype(int)
        
    def set_y_true(self, y_true):
        self.y_true = y_true
    
    
    def check_prediction_attributes(self):
        if not hasattr(self, 'y_true'):
            raise Exception('Please add y_true attribute using set_y_true(y_true) function')
        if not hasattr(self, 'y_proba'):
            raise Exception('Please add y_proba attribute using compute_prediction(X) function')
        if not hasattr(self, 'y_preds'):
            raise Exception('Please add y_preds attribute using compute_prediction(X) function')
    
    def show_confusion_matrix(self, fig=None, ncols=2, index=1):
        self.check_prediction_attributes()
        matrix = confusion_matrix(self.y_true, self.y_preds)
        
        if fig == None:
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(1, 1, 1)
        else:
            ax = fig.add_subplot(1, ncols, index)
        
        sns.heatmap(matrix, cmap='Blues', fmt='d', annot=True)
        plt.title('Confusion Matrix')
        if fig == None:
            plt.show()
        
        
    def show_accuracy(self):
        self.check_prediction_attributes()
        accuracy = accuracy_score(self.y_true, self.y_preds)
        display(Markdown('#### Accuracy of the model :'))
        print(accuracy)
        
        
    def show_f1_score(self):
        self.check_prediction_attributes()
        f1 = f1_score(self.y_true, self.y_preds)
        display(Markdown('#### F1 score of the model :'))
        print(f1)
        
    
    def show_roc_curve(self, fig=None, ncols=2, index=1):
        self.check_prediction_attributes()
        fpr, tpr, threshold = roc_curve(self.y_true, self.y_proba)
        roc_auc = auc(fpr, tpr)
        
        if fig == None:
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(1, 1, 1)
        else:
            ax = fig.add_subplot(1, ncols, index)
            
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic curve')
        plt.legend(loc="lower right")
        if fig == None:
            plt.show()
        
    
    def show_performance(self, with_roc=True):
        self.check_prediction_attributes()
        
        display(Markdown('## Overall model performance'))
        self.show_accuracy()
        self.show_f1_score()
        if with_roc:
            fig = plt.figure(figsize=(15, 6))
            self.show_confusion_matrix(fig=fig, ncols=2, index=1)
            self.show_roc_curve(fig=fig, ncols=2, index=2)
            plt.show()
        else:
            self.show_confusion_matrix()
            
            
    def compute_threshold_prediction(self, X, start=0, end=1, step=0.05):
        threshold = start
        threshold_df = pd.DataFrame()
        
        while threshold <= end:
            threshold = round(threshold,2)

            proba, preds = self.predict(X, threshold=threshold)
            threshold_df[threshold] = preds
            
            threshold += step
            
        self.threshold_df = threshold_df
        
    
    def compare_threshold_predictions(self):
        if not hasattr(self, 'threshold_df'):
            raise Exception('threshold_df not initialized : execute compute_threshold_prediction() first')
        if not hasattr(self, 'y_true'):
            raise Exception('Please add y_true attribute using set_y_true(y_true) function')
        
        df = list()
        for threshold in self.threshold_df:            
            accuracy = accuracy_score(self.y_true, self.threshold_df[threshold])
            f1 = f1_score(self.y_true, self.threshold_df[threshold])
            precision = precision_score(self.y_true, self.threshold_df[threshold])
            recall = recall_score(self.y_true, self.threshold_df[threshold])
            df.append([accuracy, f1, precision, recall])
            
        df = pd.DataFrame(df, 
                          columns=['accuracy', 'f1_score', 'precision_score', 'recall_score'], 
                          index=self.threshold_df.columns
                         )
        
        scores = df.columns
        display(Markdown('## Proba threshold comparison for accuracy, f1 score, precision & recall'))
        fig = plt.figure(figsize=(15, 10))
        
        for i in range(0,4):
            ax = fig.add_subplot(2, 2, i+1)
            lw = 2
            ax.plot(df.index, df[scores[i]], color='navy', lw=lw, label=scores[i])
            ax.set_xlabel('threshold')
            ax.set_ylabel(scores[i])
            ax.legend(loc="lower right")
        
        plt.show()