package BACKUP;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.*;
import weka.core.Instances;

public class weka_learning {

	public static void main(String[] args) throws Exception {
		
		BufferedReader breader = null;
		breader = new BufferedReader(new FileReader("C:/Users/User/Dropbox/weka-3-6-13/Gao-data/43-category-12-01-01-06/43_category_traini_completed.arff"));
		//breader = new BufferedReader(new FileReader("/home/gao/Downloads/weka-3-6-13/data/iris.2D.arff"));	
		
		Instances train = new Instances(breader);
		train.setClassIndex(train.numAttributes()-1);
		
		breader.close();
		
		NaiveBayes nB = new NaiveBayes();
		nB.buildClassifier(train);
		Evaluation eval = new Evaluation(train);
		//eval.crossValidateModel(nB, train, 10, new Random(1));
		eval.crossValidateModel(nB, train, 10, new Random(1), new Object[]{ }); //new Random(1) the '1' is the seed 
		
		System.out.println(eval.toSummaryString("\nResults\n=======\n", true));
		System.out.println(eval.fMeasure(1)+ " "+ eval.precision(1)+ " " + eval.recall(1));

	}

}
