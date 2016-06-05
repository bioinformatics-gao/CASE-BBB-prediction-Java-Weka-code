package BACKUP;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.math3.stat.inference.TestUtils;
import org.apache.commons.math3.stat.inference.TestUtils.*;

public class P_Value {
	
	static public void main(String[] args) throws NumberFormatException, IOException{
		double[] Ax=P_Value.read_file("Test_Accuracy_9Chemical.txt");
		double[] Ay=P_Value.read_file("Test_Accuracy_43CNS.txt");
		double[] Az=P_Value.read_file("Test_Accuracy_9Chemical_43CNS.txt");
		double[] Ux=P_Value.read_file("Test_AUC_9Chemical.txt");
		double[] Uy=P_Value.read_file("Test_AUC_43CNS.txt");
		double[] Uz=P_Value.read_file("Test_AUC_9Chemical_43CNS.txt");
		System.out.println(TestUtils.tTest(Ax,Ay));
		System.out.println(TestUtils.tTest(Ax,Az));
		System.out.println(TestUtils.tTest(Ay,Az));
		System.out.println(TestUtils.tTest(Ux,Uy));
		System.out.println(TestUtils.tTest(Ux,Uz));
		System.out.println(TestUtils.tTest(Uy,Uz));
	}


	static public double[] read_file(String file) throws NumberFormatException, IOException{
		double[] double_num=new double[1000];
		BufferedReader r=new BufferedReader(new FileReader(file));
		String line;
		int cnt=0;
		while((line=r.readLine())!=null){
			double num=Double.parseDouble(line);
			double_num[cnt]=num;
			cnt++;
		}
		r.close();
		
		return double_num;
	}
}
