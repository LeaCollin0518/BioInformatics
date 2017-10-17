/*
 * @author Lea Collin
 */
import weka.core.Instance;
import weka.core.Instances;
import weka.classifiers.Classifier;
import weka.classifiers.AbstractClassifier;
import weka.core.converters.ArffLoader;
import weka.attributeSelection.*;
import java.io.File;
import java.io.PrintWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.Properties;
import java.sql.*;

public class Driver {
	
	public static void main(String [] args) throws Exception {
		
		//make these strings be taken in as program arguments
		String trainingFile = args[0];
		
		String testingFile = args[1];
		
		String outputDir = args[2];
		
		//reading the files and getting all the instances of each one
		Instances instancesTrain = fileReader(trainingFile);
		Instances instancesTest = fileReader(testingFile);
		
		String reduce = "no";
		
		if(reduce.equals("yes")){
			//selecting most relevant attributes
			Instances [] reduced = attributeSelector(instancesTrain, instancesTest);
			
			instancesTrain = reduced[0];
			
			instancesTest = reduced[1];
		}
		
		//what attribute do we want to predict
		String classAttribute = "Stage";
		
		String dbUsr = args[3];
		String dbPwd = args[4];
		String dbConfig = args[5];
		
		connectToDatabase(dbUsr, dbPwd, dbConfig);
		
		/*String [] options = null;
		
		String [] classifiers = {"ZeroR", "J48", "RandomForest", "RandomTree", "NaiveBayes"};
		
		//need to keep track of the precision of different algorithms
		HashMap <String, Double> precisionVals = new HashMap <String, Double>();
		double maxPrecision = 0.0;
		String bestMethod = "";
		
		//running each different classifier, population HashMap to store each precision value
		for(int i = 0; i < classifiers.length; i++) {
			String toTest = classifiers[i];
			precisionVals.put(toTest, predict(instancesTrain, instancesTest, toTest, options, classAttribute, outputDir));
		}
		
		for(HashMap.Entry <String, Double> entry : precisionVals.entrySet()) {
			
			if(entry.getValue() > maxPrecision) {
				maxPrecision = entry.getValue();
				bestMethod = entry.getKey();
			}
		}
		
		System.out.println("Best Method: " + bestMethod + ", Precision: " + maxPrecision);*/
	}
		
	public static Instances fileReader(String input) throws IOException {

		File inputFile = new File(input);
		ArffLoader atf = new ArffLoader();
		atf.setFile(inputFile);
		Instances data = atf.getDataSet();
		
		return data;
		
	}
	
	public static Double predict(Instances train, Instances test, String classifierName, 
			String [] options, String classAttribute, String outputDir) throws Exception {
		
		//set the Class (what we want to predict)
		test.setClass(test.attribute(classAttribute));
		
		//setting the train class index to be the same as the testing class index
		train.setClassIndex(test.classIndex());
		
		double numInst = test.numInstances(), correct = 0.0f;
		
		Classifier m_classifier = AbstractClassifier.forName(classifierName, options);
		
		//building the model
		m_classifier.buildClassifier(train);
		
		
		String outputFile = outputDir + classifierName + "Predicted.csv";
		
		//writing the header to the output csv
		File output = new File(outputFile);
		PrintWriter pw = new PrintWriter(output);
        StringBuilder sb = new StringBuilder();
        sb.append("Instance");
        sb.append(',');
        sb.append("Actual");
        sb.append(',');
        sb.append("Predicted");
        sb.append('\n');
		
		for(int i = 0;i < numInst; i++){
			
			Instance current = test.instance(i);
			
			Instance temp = (Instance)current.copy();
			
			//attributes are given as array positions, getting the string value
			String actualVal = current.stringValue(test.classIndex());
			
			//getting the predicted value of the class attribute of this instance
			double predicted = m_classifier.classifyInstance(test.instance(i));
			
			//setting this value to the temp class attribute
			temp.setValue(test.classIndex(), predicted);
			
			//getting the string value
			String predictedVal = temp.stringValue(temp.classIndex());
			
			sb.append((i+1));
			sb.append(',');
			sb.append(actualVal);
			sb.append(',');
			sb.append(predictedVal);
			sb.append('\n');
			
			
			if(predictedVal.equals(actualVal)) {
		
				correct++;
			}
		}
		
		sb.append('\n');
		pw.write(sb.toString());
        pw.close();
        
        Double precision = 100*correct/numInst;
        
        return precision;
	}
	
	public static Instances [] attributeSelector(Instances train, Instances test) throws Exception {
		
		AttributeSelection selector = new AttributeSelection();
		
		//CfsSubsetEval evaluator = new CfsSubsetEval();
		//BestFirst search = new BestFirst();
		
		selector.SelectAttributes(train);
		
		Instances trainTemp = selector.reduceDimensionality(train);
		Instances trainTest = selector.reduceDimensionality(test);
		
		Instances [] reduced = {trainTemp, trainTest};

		return reduced;
	}
	
	private static void connectToDatabase(String usrDB, String passwordDB, String conDB) throws SQLException {
	    try {
	    	Class.forName("org.postgresql.Driver");
	    	Connection conn = DriverManager.getConnection(conDB, usrDB, passwordDB);
		
			Statement stmt = conn.createStatement();
			ResultSet trainingSet = stmt.executeQuery("SELECT i.camera, i.fdate, o.area, o.perimeter, "
					+ "o.circularity, o.compactness, o.major, "
					+ "o.minor, o.eccentricity, o.hisgreypeak, "
					+ "o.q1grey, o.q2grey, o.q3grey, o.q1r, o.q2r, o.q3r, o.q1g, o.q2g, o.q3g, o.q1b, o.q2b, o.q3b,"
					+ " s.growthcond, d.das, "
					+ "CASE WHEN ( d.das <= 17 ) THEN 'Stage 1' "
					+ "WHEN ( d.das > 18 AND d.das <= 25 ) THEN 'Stage 2' "
					+ "WHEN ( d.das > 25 AND d.das <= 32 ) THEN 'Stage 3' "
					+ "WHEN ( d.das > 33 AND d.das <= 40 ) THEN 'Stage 4' "
					+ "WHEN ( d.das > 40 AND d.das <= 47 ) THEN 'Stage 5' "
					+ "ELSE 'Stage 6' END Stage "
					+ "FROM imageev AS i, imgobjectev AS o, soyidentification AS s, dasplusev AS d "
					+ "WHERE i.assayid = o.assayid "
					+ "AND i.imgid = o.imgid "
					+ "AND s.barcode = ( CAST( i.barcode AS INTEGER ) ) "
					+ "AND i.assayid = d.assayid "
					+ "AND i.fdate = d.fdate "
					+ "AND i.set = d.set "
					+ "AND s.line = 1 "
					+ "AND i.camera = 'vis-side-1-0' "
					+ "AND i.set = '3'");
			while(trainingSet.next()) {
				Double area = trainingSet.getDouble("area");
				Double perimeter = trainingSet.getDouble("perimeter");
				Double circularity = trainingSet.getDouble("circularity");
				Double compactness = trainingSet.getDouble("compactness");
				Double major = trainingSet.getDouble("major");
				Double minor = trainingSet.getDouble("minor");
				Double eccentricity = trainingSet.getDouble("eccentricity");
				Double hisgreypeak = trainingSet.getDouble("hisgreypeak");
				Double q1grey = trainingSet.getDouble("q1grey");
				Double q2grey = trainingSet.getDouble("q2grey");
				Double q3grey = trainingSet.getDouble("q3grey");
				Double q1r = trainingSet.getDouble("q1r");
				Double q2r = trainingSet.getDouble("q2r");
				Double q3r = trainingSet.getDouble("q3r");
				Double q1g = trainingSet.getDouble("q1g");
				Double q2g = trainingSet.getDouble("q2g");
				Double q3g = trainingSet.getDouble("q3g");
				Double q1b = trainingSet.getDouble("q1b");
				Double q2b = trainingSet.getDouble("q2b");
				Double q3b = trainingSet.getDouble("q3b");
				String stage = trainingSet.getString("Stage");
				
				System.out.println("Area: " + circularity + ", Perimeter: " + stage);
			}
			trainingSet.close();
			conn.close();
		} 
	    catch (ClassNotFoundException e) {

			System.out.println("Where is your PostgreSQL JDBC Driver? "
					+ "Include in your library path!");
			e.printStackTrace();
			return;

		}
	}	
}