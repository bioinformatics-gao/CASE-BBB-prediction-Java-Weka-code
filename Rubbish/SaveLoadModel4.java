package Rubbish;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;

import java.text.DecimalFormat;


public class SaveLoadModel4 {
	public static void main(String args[]) throws Exception{
		DecimalFormat f = new DecimalFormat("##.000");
	    
		for(int num=1; num<=10; num++){
			
			//load training data
			DataSource source = new DataSource("C:/Users/User/Dropbox/Maching-Learning-Weka/2016-02-28-data/2002-Doniger-9-features-and-biology/chemical-and-43CNS-training.arff");
			Instances trainDataset = source.getDataSet();	
			trainDataset.setClassIndex(trainDataset.numAttributes()-1);
			SMO smo = new SMO();
			smo.buildClassifier(trainDataset);
			System.out.println(smo);
			weka.core.SerializationHelper.write("my_smo_model"+ num +".model", smo);
			
			Evaluation eval = new Evaluation(trainDataset);
			eval.evaluateModel(smo,trainDataset);
			
			System.out.println("Accuracy:  "+ f.format((1-eval.errorRate())));
			System.out.println("AUC = " + f.format(eval.areaUnderROC(1)));
			System.out.println("Precision:  "+ f.format(eval.precision(1)));
			System.out.println("Recall:  "+f.format(eval.recall(1)));
			System.out.println("Fmearure:  "+ f.format(eval.fMeasure(1)));
			System.out.println(eval.toMatrixString("=== Confusion matrix "));
	
			SMO smo2 = (SMO) weka.core.SerializationHelper.read("my_smo_model"+ num +".model");
			DataSource source1 = new DataSource("C:/Users/User/Dropbox/Maching-Learning-Weka/2016-02-28-data/2002-Doniger-9-features-and-biology/chemical-and-43CNS-validation.arff");
			Instances testDataset = source1.getDataSet();	
			testDataset.setClassIndex(testDataset.numAttributes()-1);
			
			double actualValue = testDataset.instance(31).classValue();
			Instance newInst = testDataset.instance(31);
			double predSMO = smo2.classifyInstance(newInst);
			System.out.println(actualValue+", "+predSMO);
	
			Evaluation eval2 = new Evaluation(testDataset);
			eval2.evaluateModel(smo2,testDataset);
			
			System.out.println("Accuracy:  "+ f.format((1-eval2.errorRate())));
			System.out.println("AUC = " + f.format(eval2.areaUnderROC(1)));
			System.out.println("Precision:  "+ f.format(eval2.precision(1)));
			System.out.println("Recall:  "+f.format(eval2.recall(1)));
			System.out.println("Fmearure:  "+ f.format(eval2.fMeasure(1)));
			System.out.println(eval2.toMatrixString("=== Confusion matrix "));
		}
	}	
}
