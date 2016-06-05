package BAK0304;
import weka.core.Instance;
import weka.filters.unsupervised.instance.Resample;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;

import java.text.DecimalFormat;

public class SaveLoadModel_for_gao_curqated_data {

	public static void main(String args[]) throws Exception{
		DecimalFormat f = new DecimalFormat("##.000");
	    Evaluation eval1[] = new Evaluation[1000] ;
	    Evaluation eval2[] = new Evaluation[1000];
	    
		for(int num=0; num< 1000; num++){
			
			// Load data  
			DataSource source = new DataSource("C:/Users/User/Dropbox/Maching-Learning-Weka/2016-02-28-data/Zhen-curated-data/trainig_CNS_endo_eye-manully-added-01-25-1.csv");
			
			Instances data = source.getDataSet();
			//Instances inst = getInputFormat();
			
			if (data.classIndex() == -1)   data.setClassIndex(data.numAttributes() - 1);

			Resample filter = new Resample();
	        filter.setRandomSeed(num);
	        filter.setInvertSelection(true);
	        filter.setSampleSizePercent(70);
	        filter.setInputFormat(data);
			// apply filter for test data here
	        Instances trainDataset = Filter.useFilter(data, filter);
			
			//  prepare and apply filter for training data here
			filter.setInvertSelection(true);     // invert the selection to get other data 
			Instances testDataset = Filter.useFilter(data, filter);

			SMO smo = new SMO();
			smo.buildClassifier(trainDataset);
			System.out.println(smo);
			weka.core.SerializationHelper.write("my_smo_model"+ num +".model", smo);
			
			eval1[num] = new Evaluation(trainDataset);
			eval1[num].evaluateModel(smo,trainDataset);
			
			SMO smo2 = (SMO) weka.core.SerializationHelper.read("my_smo_model"+ num +".model");
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
		for(int num=0; num< 1000; num++){
			Total_Accuracy1 +=  1-eval1[num].errorRate();
			Total_AUC1 += eval1[num].areaUnderROC(1);
			Total_fMeasure1 += eval1[num].fMeasure(1);
			Total_Accuracy2 +=  1-eval2[num].errorRate();
			Total_AUC2 += eval2[num].areaUnderROC(1);
			Total_fMeasure2 += eval2[num].fMeasure(1);
		}
		
		System.out.println("Train Average Accuracy:  "+ f.format(Total_Accuracy1/1000));
		System.out.println("Train Average AUC = " + f.format(Total_AUC1/1000));
		System.out.println("Train Average Fmearure:  "+ f.format(Total_fMeasure1/1000));
		System.out.println("Test Average Accuracy:  "+ f.format(Total_Accuracy2/1000));
		System.out.println("Test Average AUC = " + f.format(Total_AUC2/1000));
		System.out.println("Test Average Fmearure:  "+ f.format(Total_fMeasure2/1000));
		
	}
}
