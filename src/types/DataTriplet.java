package types;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.ArrayList;

import pipes.Target2CoNLLLabel;
import pipes.Target2SchemeBLabel;
import pipes.Target2TopLevelLabel;
import cc.mallet.classify.Classification;
import cc.mallet.pipe.Csv2FeatureVector;
import cc.mallet.pipe.Pipe;
import cc.mallet.pipe.SerialPipes;
import cc.mallet.pipe.Target2Label;
import cc.mallet.pipe.iterator.CsvIterator;
import cc.mallet.types.Alphabet;
import cc.mallet.types.InstanceList;

/*
 * Holds a triplet of training, development, test sets that use the same set of features
 */
public class DataTriplet {
	public static final String DATA_PATTERN = "(\\w+)\\t([^\\t]+)\\t(.+)";
	
	private InstanceList trainingSet;
	private InstanceList devSet;
	private InstanceList testSet;	
	private String trainingFileName;
	private String devFileName;
	private String testFileName;
	
	public DataTriplet(String trainingFileName, String devFileName, String testFileName) {
		this.setTrainingFileName(trainingFileName);
		this.setDevFileName(devFileName);
		this.setTestFileName(testFileName);
	}
	
	/*
	 * importData has to be called first. The label scheme has to be explicitly selected.
	 */
	public InstanceList getTrainingSet() throws FileNotFoundException { 
		return trainingSet; 
	}

	/*
	 * importData has to be called first. The label scheme has to be explicitly selected.
	 */
	public InstanceList getDevSet() throws FileNotFoundException { 
		return devSet; 
	}

	/*
	 * importData has to be called first. The label scheme has to be explicitly selected.
	 */
	public InstanceList getTestSet() throws FileNotFoundException { 
		return testSet; 
	}
	
	public void importData(String trainingFileName, 
			String devFileName, String testFileName, LabelType labelType) throws FileNotFoundException{

		System.out.println("Reading training set :" + trainingFileName);
		trainingSet = importData(trainingFileName, labelType);
		System.out.println("Using " + trainingSet.getAlphabet().size() + " features.");
		
		System.out.println("Reading dev set :" + devFileName);
		devSet = importData(devFileName, trainingSet.getPipe());
		
		System.out.println("Reading test set :" + testFileName);
		testSet = importData(testFileName, trainingSet.getPipe());

	}
	public void importData(LabelType labelType) throws FileNotFoundException{
		importData(trainingFileName, devFileName, testFileName, labelType);
	}

	public void importData() throws FileNotFoundException{
		importData(trainingFileName, devFileName, testFileName, null);
	}

	
	public static InstanceList importData(String fileName, Pipe pipe) throws FileNotFoundException {
		return importData(fileName, null, pipe, null);
	}
	public static InstanceList importData(String dataFile, LabelType labelType) throws FileNotFoundException {
		return importData(dataFile, labelType, null);
	}

	public static InstanceList importData(String dataFile, LabelType labelType,
			Alphabet alphabet) throws FileNotFoundException {
		return importData(dataFile, labelType, null, alphabet);
	}	
	
	public static Pipe getPipe(LabelType labelType) {
		ArrayList<Pipe> pipes = new ArrayList<Pipe>();
		pipes.add(new Csv2FeatureVector());
		if (labelType == null){
			pipes.add(new Target2Label());
		}else {
			switch (labelType){
			case CONLL:
				pipes.add(new Target2CoNLLLabel());
				break;
			case TOP_LEVEL:
				pipes.add(new Target2TopLevelLabel());
				break;
			case SCHEME_B:
				pipes.add(new Target2SchemeBLabel());
				break;
			}
		}
		Pipe pipe = new SerialPipes(pipes);
		pipe.setTargetProcessing(true);
		return pipe;
	}

	/* Import data (used for multiple datasets settings)
	 * 
	 * The pipe from the first dataset should be passed into the method to create
	 * consistent Alphabets across datasets that are being imported
	 */
	public static InstanceList importData(String fileName, LabelType labelType, Pipe pipe, Alphabet alphabet) throws FileNotFoundException {
		if (pipe == null) {
			pipe = getPipe(labelType);
		}
		if (alphabet != null) pipe.setDataAlphabet(alphabet);

		InstanceList data = new InstanceList(pipe); 
		data.addThruPipe(new CsvIterator(new FileReader(fileName), DATA_PATTERN, 3, 2, 1));
		
		int index = 0;
		while (index < data.size()){
			if (data.get(index).getTarget().toString().equals("**NULL**")) {
				data.remove(index);
			} else {
				index++;
			}
		}
		
		return data;
	}
	
	public static String[] getStringLabels(InstanceList data) {
		int numInstances = data.size();
		String[] labels = new String[numInstances];
		for (int i = 0; i < numInstances; i++ ){
			labels[i] = (String)data.get(i).getLabeling().getBestLabel().getEntry();
		}
		return labels;
	}
	
	public static String[] getStringLabels(ArrayList<Classification> data) {
		int numInstances = data.size();
		String[] labels = new String[numInstances];
		for (int i = 0; i < numInstances; i++ ){
			labels[i] = (String)data.get(i).getLabeling().getBestLabel().getEntry();
		}
		return labels;
	}
	

	public String getTrainingFileName() {
		return trainingFileName;
	}

	public void setTrainingFileName(String trainingFileName) {
		this.trainingFileName = trainingFileName;
	}

	public String getDevFileName() {
		return devFileName;
	}

	public void setDevFileName(String devFileName) {
		this.devFileName = devFileName;
	}

	public String getTestFileName() {
		return testFileName;
	}

	public void setTestFileName(String testFileName) {
		this.testFileName = testFileName;
	}

}
