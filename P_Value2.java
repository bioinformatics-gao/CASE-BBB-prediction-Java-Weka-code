import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.math3.stat.inference.TestUtils;
import org.apache.commons.math3.stat.inference.TestUtils.*;

public class P_Value2 {
	
	static public void main(String[] args) throws NumberFormatException, IOException{
		double[] Aa=P_Value2.read_file("Test_Accuracy_0306_Doniger_Chemcals.txt");
		double[] Af=P_Value2.read_file("Test_Accuracy_160304_Zhen_43CNS_Indications.txt");
		double[] Ag=P_Value2.read_file("Test_Accuracy_Doniger_43CNS_Chemcals_Indications.txt");
		double[] Ua=P_Value2.read_file("Test_AUC_0306_Doniger_Chemcals.txt");
		double[] Uf=P_Value2.read_file("Test_AUC_160304_Zhen_43CNS_Indications.txt");
		double[] Ug=P_Value2.read_file("Test_AUC_Doniger_43CNS_Chemcals_Indications.txt");
		

		System.out.println(TestUtils.tTest(Aa,Af));
		System.out.println(TestUtils.tTest(Aa,Ag));
		System.out.println(TestUtils.tTest(Af,Ag));
		System.out.println(TestUtils.tTest(Ua,Uf));
		System.out.println(TestUtils.tTest(Ua,Ug));
		System.out.println(TestUtils.tTest(Uf,Ug));
		
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
