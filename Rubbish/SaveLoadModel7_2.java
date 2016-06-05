package Rubbish;
import weka.filters.unsupervised.instance.Resample;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import java.io.*;
import java.text.DecimalFormat;

public class SaveLoadModel7_2 {

	public static void main(String args[]) throws Exception{
		DecimalFormat f = new DecimalFormat("##.000");
	    Evaluation eval1[] = new Evaluation[1000] ;
	    Evaluation eval2[] = new Evaluation[1000];
	    
		for(int num=0; num< 1000; num++){
			
			// Load data  
			DataSource source = new DataSource("C:/Users/User/Dropbox/Maching-Learning-Weka/2016-02-28-data/2002-Doniger-9-features-and-biology/2002-Doniger-chemical.csv");
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
			weka.core.SerializationHelper.write("9Chem_smo_model"+ num +".model", smo);
			
			eval1[num] = new Evaluation(trainDataset);
			eval1[num].evaluateModel(smo,trainDataset);
			
			SMO smo2 = (SMO) weka.core.SerializationHelper.read("9Chem_smo_model"+ num +".model");
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
		double[] Test_Accuracy_9Chemical_43CNS = new double[1000];  
		double[] Test_AUC_9Chemical_43CNS = new double[1000]; 
		for(int num=0; num< 1000; num++){
			Total_Accuracy1 +=  1-eval1[num].errorRate();
			Total_AUC1 += eval1[num].areaUnderROC(1);
			Total_fMeasure1 += eval1[num].fMeasure(1);
			Test_Accuracy_9Chemical_43CNS[num]= 1-eval2[num].errorRate();
			Total_Accuracy2 +=  1-eval2[num].errorRate();
			Test_AUC_9Chemical_43CNS[num]=eval2[num].areaUnderROC(1);
			Total_AUC2 += eval2[num].areaUnderROC(1);
			Total_fMeasure2 += eval2[num].fMeasure(1);
		}
		
		write("Test_Accuracy_9Chemical.txt", Test_Accuracy_9Chemical_43CNS);
		write("Test_AUC_9Chemical.txt", Test_AUC_9Chemical_43CNS);
		
		System.out.println("Train Average Accuracy:  "+ f.format(Total_Accuracy1/1000));
		System.out.println("Train Average AUC = " + f.format(Total_AUC1/1000));
		System.out.println("Train Average Fmearure:  "+ f.format(Total_fMeasure1/1000));
		System.out.println("Test Average Accuracy:  "+ f.format(Total_Accuracy2/1000));
		System.out.println("Test Average AUC = " + f.format(Total_AUC2/1000));
		System.out.println("Test Average Fmearure:  "+ f.format(Total_fMeasure2/1000));
		
	}
	
	public static void write (String filename, double[] x) throws IOException{
			  BufferedWriter outputWriter = null;
			  outputWriter = new BufferedWriter(new FileWriter(filename));
			  for (int i = 0; i < x.length; i++) {
			    outputWriter.write(Double.toString(x[i]));
			    outputWriter.newLine();
			  }
			  outputWriter.flush();  
			  outputWriter.close();  
	}
	
}
