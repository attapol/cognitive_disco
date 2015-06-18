package types;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.ArrayList;

import cc.mallet.pipe.Csv2FeatureVector;
import cc.mallet.pipe.Pipe;
import cc.mallet.pipe.SerialPipes;
import cc.mallet.pipe.Target2Label;
import cc.mallet.pipe.iterator.CsvIterator;
import cc.mallet.types.InstanceList;

public class DataTriplet {
	public static final String DATA_PATTERN = "(\\w+)\\t([^\\t]+)\\t(.+)";
	
	private InstanceList trainingSet;
	private InstanceList devSet;
	private InstanceList testSet;	
	
	public DataTriplet(String trainingFileName, 
			String devFileName, String testFileName) throws FileNotFoundException{
		importData(trainingFileName, devFileName, testFileName);
	}
	
	public InstanceList getTrainingSet() { return trainingSet; }
	public InstanceList getDevSet() { return devSet; }
	public InstanceList getTestSet() { return testSet; }
	
	public void importData(String trainingFileName, 
			String devFileName, String testFileName) throws FileNotFoundException{
		System.out.println("Reading training set :" + trainingFileName);
		trainingSet = importData(trainingFileName);
		System.out.println("Using " + trainingSet.getAlphabet().size() + " features.");
		
		System.out.println("Reading dev set :" + devFileName);
		devSet = importData(devFileName, trainingSet.getPipe());
		
		System.out.println("Reading test set :" + testFileName);
		testSet = importData(testFileName, trainingSet.getPipe());
	}
	
	public static InstanceList importData(String fileName) throws FileNotFoundException {
		return importData(fileName, null);
	}
	public static InstanceList importData(String fileName, Pipe pipe) throws FileNotFoundException {
		/* Import data (used for multiple datasets settings)
		 * 
		 * The pipe from the first dataset should be passed into the method to create
		 * consistent Alphabets across datasets that are being imported
		 */
		if (pipe == null) {
			ArrayList<Pipe> pipes = new ArrayList<Pipe>();
			pipes.add(new Csv2FeatureVector());
			pipes.add(new Target2Label());
			pipe = new SerialPipes(pipes);
			pipe.setTargetProcessing(true);
		}
		
		InstanceList data = new InstanceList(pipe);
		data.addThruPipe(new CsvIterator(new FileReader(fileName), DATA_PATTERN, 3, 2, 1));
		
		return data;
	}	
}
