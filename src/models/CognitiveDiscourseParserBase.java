package models;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;

import org.json.JSONException;

import types.DataTriplet;
import types.DimensionMapper;
import types.SimpleConfusionMatrix;

public abstract class CognitiveDiscourseParserBase {

	protected DimensionMapper dm;
	protected DataTriplet[] data;
	protected DataTriplet originalData;
	
	public CognitiveDiscourseParserBase(String experimentName, String mappingJson, String trainingDir, String devDir, String testDir) throws JSONException, IOException{
		dm = new DimensionMapper(mappingJson);
		System.out.println(dm.toString());
		data = new DataTriplet[dm.numDimensions()];
		ArrayList<String> dimensions = dm.getDimensions();
		for (int i = 0; i < dimensions.size(); i++){
			String dimension = dimensions.get(i);
			data[i] = loadData(experimentName, dimension, trainingDir, devDir, testDir);
		}
		originalData = loadOriginalData(experimentName, trainingDir, devDir, testDir);
	}
	
	private DataTriplet loadOriginalData(String experimentName,
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
		System.out.println(cm.toString());
	}
	
	abstract public void trainTest();
	
	/*
	 * Supposedly, it will call the Python featurizer and write features to file.
	 * Then classify the data 
	 */
	abstract public void classify(String dataDir);
}
