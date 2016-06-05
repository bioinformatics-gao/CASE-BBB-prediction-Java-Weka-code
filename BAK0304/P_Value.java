package BAK0304;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.math3.stat.inference.TestUtils;
import org.apache.commons.math3.stat.inference.TestUtils.*;

public class P_Value {
	
	static public void main(String[] args) throws NumberFormatException, IOException{
		double[] Aa=P_Value.read_file("Test_Accuracy9Chem.txt");
		double[] Ab=P_Value.read_file("Test_Accuracy_43CNS.txt");
		double[] Ac=P_Value.read_file("Test_Accuracy_Indication.txt");
		double[] Ad=P_Value.read_file("Test_Accuracy_43CNS_9Chem.txt");
		double[] Ae=P_Value.read_file("Test_Accuracy_9Chem_Indication.txt");
		double[] Af=P_Value.read_file("Test_Accuracy_43CNS_Indication.txt");
		double[] Ag=P_Value.read_file("Test_Accuracy_43CNS_9Chem_Indication.txt");
		
		double[] Ua=P_Value.read_file("Test_AUC9Chem.txt");
		double[] Ub=P_Value.read_file("Test_AUC_43CNS.txt");
		double[] Uc=P_Value.read_file("Test_AUC_Indication.txt");
		double[] Ud=P_Value.read_file("Test_AUC_43CNS_9Chem.txt");
		double[] Ue=P_Value.read_file("Test_AUC_9Chem_Indication.txt");
		double[] Uf=P_Value.read_file("Test_AUC_43CNS_Indication.txt");
		double[] Ug=P_Value.read_file("Test_AUC_43CNS_9Chem_Indication.txt");
		
		System.out.println(TestUtils.tTest(Aa,Ab));
		System.out.println(TestUtils.tTest(Aa,Ac));
		System.out.println(TestUtils.tTest(Aa,Ad));
		System.out.println(TestUtils.tTest(Aa,Ae));
		System.out.println(TestUtils.tTest(Aa,Af));
		System.out.println(TestUtils.tTest(Aa,Ag));
		System.out.println(TestUtils.tTest(Ab,Ac));
		System.out.println(TestUtils.tTest(Ab,Ad));
		System.out.println(TestUtils.tTest(Ab,Ae));
		System.out.println(TestUtils.tTest(Ab,Af));
		System.out.println(TestUtils.tTest(Ab,Ag));
		System.out.println(TestUtils.tTest(Ac,Ad));
		System.out.println(TestUtils.tTest(Ac,Ae));
		System.out.println(TestUtils.tTest(Ac,Af));
		System.out.println(TestUtils.tTest(Ac,Ag));
		System.out.println(TestUtils.tTest(Ad,Ae));
		System.out.println(TestUtils.tTest(Ad,Af));
		System.out.println(TestUtils.tTest(Ad,Ag));
		System.out.println(TestUtils.tTest(Ae,Af));
		System.out.println(TestUtils.tTest(Ae,Ag));
		System.out.println(TestUtils.tTest(Af,Ag));
		
		System.out.println(TestUtils.tTest(Ua,Ub));
		System.out.println(TestUtils.tTest(Ua,Uc));
		System.out.println(TestUtils.tTest(Ua,Ud));
		System.out.println(TestUtils.tTest(Ua,Ue));
		System.out.println(TestUtils.tTest(Ua,Uf));
		System.out.println(TestUtils.tTest(Ua,Ug));
		System.out.println(TestUtils.tTest(Ub,Uc));
		System.out.println(TestUtils.tTest(Ub,Ud));
		System.out.println(TestUtils.tTest(Ub,Ue));
		System.out.println(TestUtils.tTest(Ub,Uf));
		System.out.println(TestUtils.tTest(Ub,Ug));
		System.out.println(TestUtils.tTest(Uc,Ud));
		System.out.println(TestUtils.tTest(Uc,Ue));
		System.out.println(TestUtils.tTest(Uc,Uf));
		System.out.println(TestUtils.tTest(Uc,Ug));
		System.out.println(TestUtils.tTest(Ud,Ue));
		System.out.println(TestUtils.tTest(Ud,Uf));
		System.out.println(TestUtils.tTest(Ud,Ug));
		System.out.println(TestUtils.tTest(Ue,Uf));
		System.out.println(TestUtils.tTest(Ue,Ug));
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
