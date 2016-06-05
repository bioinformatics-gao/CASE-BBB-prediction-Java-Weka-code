package Rubbish;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.functions.SMO;

public class SaveLoadModel2 {
	public static void main(String args[]) throws Exception{
		
		//load training data
		DataSource source = new DataSource("C:/Users/User/Dropbox/Maching-Learning-Weka/2016-02-28-data/2002-Doniger-9-features-and-biology/chemical-and-43CNS-training.arff");
		Instances trainDataset = source.getDataSet();	
		trainDataset.setClassIndex(trainDataset.numAttributes()-1);
		SMO smo = new SMO();
		smo.buildClassifier(trainDataset);
		System.out.println(smo);
		weka.core.SerializationHelper.write("my_smo_model.model", smo);

		SMO smo2 = (SMO) weka.core.SerializationHelper.read("my_smo_model.model");
		DataSource source1 = new DataSource("C:/Users/User/Dropbox/Maching-Learning-Weka/2016-02-28-data/2002-Doniger-9-features-and-biology/chemical-and-43CNS-validation.arff");
		Instances testDataset = source1.getDataSet();	
		testDataset.setClassIndex(testDataset.numAttributes()-1);
		
		double actualValue = testDataset.instance(0).classValue();
		Instance newInst = testDataset.instance(0);
		double predSMO = smo2.classifyInstance(newInst);

		System.out.println(actualValue+", "+predSMO);
	}
}
