package BAK0304;
//import weka.core.Instance;
import weka.filters.unsupervised.instance.Resample;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
//import org.apache.commons.math3.stat.inference.TestUtils.*;
import java.io.*;
import java.text.DecimalFormat;
import org.apache.commons.math3.stat.StatUtils;
//import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.math3.util.FastMath;

public class SaveLoadModel_9Chem {

	public static void main(String args[]) throws Exception{
		DecimalFormat f = new DecimalFormat("##.000");
	    Evaluation eval1[] = new Evaluation[1000] ;
	    Evaluation eval2[] = new Evaluation[1000];
	    
		for(int num=0; num< 1000; num++){
			
			// Load data  
			DataSource source = new DataSource("C:/Users/User/Dropbox/Maching-Learning-Weka/2016-02-28-data/2002/Doniger-Chemcals.csv");
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
			weka.core.SerializationHelper.write("my_smo_model9Chem"+ num +".model", smo);
			
			eval1[num] = new Evaluation(trainDataset);
			eval1[num].evaluateModel(smo,trainDataset);
			
			SMO smo2 = (SMO) weka.core.SerializationHelper.read("my_smo_model9Chem"+ num +".model");
			testDataset.setClassIndex(testDataset.numAttributes()-1);
			
			eval2[num] = new Evaluation(testDataset);
			eval2[num].evaluateModel(smo2,testDataset);
			
		}
		
		double[] Train_Accuracy9Chem = new double[1000];  
		double[] Train_AUC9Chem = new double[1000]; 
		double[] Train_fMeasure9Chem = new double[1000]; 
		double[] Test_Accuracy9Chem = new double[1000];  
		double[] Test_AUC9Chem = new double[1000]; 
		double[] Test_fMeasure9Chem = new double[1000]; 
		
		for(int num=0; num< 1000; num++){
			Train_Accuracy9Chem[num]= 1-eval1[num].errorRate();
			Train_AUC9Chem[num]=eval1[num].areaUnderROC(1);
			Train_fMeasure9Chem[num]= eval1[num].fMeasure(1);
			
			Test_Accuracy9Chem[num]= 1-eval2[num].errorRate();
			Test_AUC9Chem[num]=eval2[num].areaUnderROC(1);
			Test_fMeasure9Chem[num]= eval2[num].fMeasure(1);
		}
		double Train_Accuracy9Chem_mean = StatUtils.mean(Train_Accuracy9Chem);
		double Train_Accuracy9Chem_std = FastMath.sqrt(StatUtils.variance(Train_Accuracy9Chem));
		double Train_AUC9Chem_mean = StatUtils.mean(Train_AUC9Chem);
		double Train_AUC9Chem_std = FastMath.sqrt(StatUtils.variance(Train_AUC9Chem));
		double Train_fMeasure9Chem_mean = StatUtils.mean(Train_fMeasure9Chem);
		double Train_fmeasure9Chem_std = FastMath.sqrt(StatUtils.variance(Train_fMeasure9Chem));
		
		double Test_Accuracy9Chem_mean = StatUtils.mean(Test_Accuracy9Chem);
		double Test_Accuracy9Chem_std = FastMath.sqrt(StatUtils.variance(Test_Accuracy9Chem));
		double Test_AUC9Chem_mean = StatUtils.mean(Test_AUC9Chem);
		double Test_AUC9Chem_std = FastMath.sqrt(StatUtils.variance(Test_AUC9Chem));
		double Test_fMeasure9Chem_mean = StatUtils.mean(Test_fMeasure9Chem);
		double Test_fmeasure9Chem_std = FastMath.sqrt(StatUtils.variance(Test_fMeasure9Chem));
		
		write("Test_Accuracy9Chem.txt", Test_Accuracy9Chem);
		write("Test_AUC9Chem.txt", Test_AUC9Chem);
		
		System.out.println("Train Accuracy Mean:  "+ f.format(Train_Accuracy9Chem_mean));
		System.out.println("Train Accuracy STD:  "+ f.format(Train_Accuracy9Chem_std));
		System.out.println("Train AUC Mean:  "+ f.format(Train_AUC9Chem_mean));
		System.out.println("Train AUC STD:  "+ f.format(Train_AUC9Chem_std));
		System.out.println("Train fMeasure Mean:  "+ f.format(Train_fMeasure9Chem_mean));
		System.out.println("Train fMeaure STD:  "+ f.format(Train_fmeasure9Chem_std));
		
		System.out.println("Test Accuracy Mean:  "+ f.format(Test_Accuracy9Chem_mean));
		System.out.println("Test Accuracy STD:  "+ f.format(Test_Accuracy9Chem_std));
		System.out.println("Test AUC Mean:  "+ f.format(Test_AUC9Chem_mean));
		System.out.println("Test AUC STD:  "+ f.format(Test_AUC9Chem_std));
		System.out.println("Test fMeasure Mean:  "+ f.format(Test_fMeasure9Chem_mean));
		System.out.println("Test fMeaure STD:  "+ f.format(Test_fmeasure9Chem_std));
		
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
