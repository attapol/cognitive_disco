package models;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;

import org.json.JSONException;

import types.DataTriplet;
import types.DimensionMapper;
import types.LabelType;
import types.ResultWriter;
import types.SimpleConfusionMatrix;

public abstract class CognitiveDiscourseParserBase {

	protected DimensionMapper dm;
	public DataTriplet[] data;
	public DataTriplet originalData;
	public ResultWriter writer;
	

	
	public CognitiveDiscourseParserBase() {
		
	}
	
	public CognitiveDiscourseParserBase(String experimentName, String mappingJson, String trainingDir, String devDir, String testDir) throws JSONException, IOException{
		dm = new DimensionMapper(mappingJson);

		writer = new ResultWriter(getModelName()+"."+experimentName+"."+dm.getMappingName());
		writer.logln(dm.toString());
		data = new DataTriplet[dm.numDimensions()];
		ArrayList<String> dimensions = dm.getDimensions();
		for (int i = 0; i < dimensions.size(); i++){
			String dimension = dimensions.get(i);
			data[i] = loadData(experimentName, dimension, trainingDir, devDir, testDir);
		}
		originalData = loadOriginalData(experimentName, trainingDir, devDir, testDir);

	}
	
	protected DataTriplet loadOriginalData(String experimentName,
			String trainingDir, String devDir, String testDir) throws FileNotFoundException {
		// TODO Auto-generated method stub
		String trainingFileName = dm.getFeatureFileName(experimentName, trainingDir);
		String devFileName = dm.getFeatureFileName(experimentName, devDir);
		String testFileName = dm.getFeatureFileName(experimentName, testDir);
		return new DataTriplet(trainingFileName, devFileName, testFileName);
	}

	private DataTriplet loadData(String experimentName, String dimension, String trainingDir, String devDir, String testDir) throws FileNotFoundException{
		String trainingFileName = dm.getFeatureFileName(experimentName, dimension, trainingDir);
		String devFileName = dm.getFeatureFileName(experimentName, dimension, devDir);
		String testFileName = dm.getFeatureFileName(experimentName, dimension, testDir);
		return new DataTriplet(trainingFileName, devFileName, testFileName);
	}
	
	public void evaluate(String[] trueLabels, String[] predictedLabels){
		SimpleConfusionMatrix cm = new SimpleConfusionMatrix(trueLabels, predictedLabels);
		writer.logln(cm.toString());
	}

	abstract public String getModelName();
	/*
	 * Train on the training set and test on the dev set 
	 * Print out the results in a confusion matrix and overall accuracy
	 */
	abstract public void trainTest() throws FileNotFoundException, JSONException;
	
	/*
	 * This will read in the feature file 
	 * and return a string array of labels
	 */
	abstract public String[] classify(String dataFile, LabelType labelType) throws FileNotFoundException;

}
