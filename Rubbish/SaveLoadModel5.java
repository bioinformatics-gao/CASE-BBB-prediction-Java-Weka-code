package Rubbish;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;

import java.text.DecimalFormat;

public class SaveLoadModel5 {

	public static void main(String args[]) throws Exception{
		DecimalFormat f = new DecimalFormat("##.000");
	    Evaluation eval1[] = new Evaluation[10] ;
	    Evaluation eval2[] = new Evaluation[10];
	    
		for(int num=0; num< 10; num++){
			
			//load training data
			DataSource source = new DataSource("C:/Users/User/Dropbox/Maching-Learning-Weka/2016-02-28-data/2002-Doniger-9-features-and-biology/chemical-and-43CNS-training.arff");
			Instances trainDataset = source.getDataSet();	
			trainDataset.setClassIndex(trainDataset.numAttributes()-1);
			SMO smo = new SMO();
			smo.buildClassifier(trainDataset);
			System.out.println(smo);
			weka.core.SerializationHelper.write("my_smo_model"+ num +".model", smo);
			
			eval1[num] = new Evaluation(trainDataset);
			eval1[num].evaluateModel(smo,trainDataset);
			
			SMO smo2 = (SMO) weka.core.SerializationHelper.read("my_smo_model"+ num +".model");
			DataSource source1 = new DataSource("C:/Users/User/Dropbox/Maching-Learning-Weka/2016-02-28-data/2002-Doniger-9-features-and-biology/chemical-and-43CNS-validation.arff");
			Instances testDataset = source1.getDataSet();	
			testDataset.setClassIndex(testDataset.numAttributes()-1);
			
			eval2[num] = new Evaluation(testDataset);
			eval2[num].evaluateModel(smo2,testDataset);
			
		}
		
		double Total_Accuracy1 = 0.0;  
		double Total_AUC1 = 0.0;  
		double Total_fMeasure1 = 0.0;  
		double Total_Accuracy2 = 0.0;  
		double Total_AUC2 = 0.0;  
		double Total_fMeasure2 = 0.0;  
		for(int num=0; num< 10; num++){
			Total_Accuracy1 +=  1-eval1[num].errorRate();
			Total_AUC1 += eval1[num].areaUnderROC(1);
			Total_fMeasure1 += eval1[num].fMeasure(1);
			Total_Accuracy2 +=  1-eval2[num].errorRate();
			Total_AUC2 += eval2[num].areaUnderROC(1);
			Total_fMeasure2 += eval2[num].fMeasure(1);
		}
		
		System.out.println("Train Average Accuracy:  "+ f.format(Total_Accuracy1/10));
		System.out.println("Train Average AUC = " + f.format(Total_AUC1/10));
		System.out.println("Train Average Fmearure:  "+ f.format(Total_fMeasure1/10));
		System.out.println("Test Average Accuracy:  "+ f.format(Total_Accuracy2/10));
		System.out.println("Test Average AUC = " + f.format(Total_AUC2/10));
		System.out.println("Test Average Fmearure:  "+ f.format(Total_fMeasure2/10));
		
	}
}
