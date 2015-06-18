package types;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.json.JSONException;
import org.json.JSONObject;

public class DimensionMapper {

	private JSONObject mapping;
	private ArrayList<String> dimensions = new ArrayList<String>();
	private ArrayList<Sense> senses = new ArrayList<Sense>();
	private String mappingName;
	private HashMap<ArrayList<String>, HashSet<Sense> > dimensionsToSense = new HashMap<ArrayList<String>, HashSet<Sense>>();
	
	
	public DimensionMapper(String mappingJson) throws IOException, JSONException{
		String jsonString = FileUtils.readFileToString(new File(mappingJson));
		mapping = new JSONObject(jsonString);
		mappingName = FilenameUtils.getBaseName(mappingJson);
		
		// collect all senses
		Iterator<String> it = mapping.keys();
		while (it.hasNext()) senses.add(new Sense(it.next()));
		
		// collect all dimension
		it = mapping.keys();
		String key = it.next();
		it = mapping.getJSONObject(key).keys();
		while (it.hasNext()) dimensions.add(it.next());
		
		for (Sense sense : senses){
			ArrayList<String> labelDimensions = new ArrayList<String>();
			for (int i = 0; i < numDimensions(); i++){
				String dimension = dimensions.get(i);
				labelDimensions.add(mapping.getJSONObject(sense.getRawSense()).getString(dimension));
			}
			if (dimensionsToSense.containsKey(labelDimensions)){
				dimensionsToSense.get(labelDimensions).add(sense);
			} else {
				HashSet<Sense> newSenseSet = new HashSet<Sense>();
				newSenseSet.add(sense);
				dimensionsToSense.put(labelDimensions, newSenseSet);
			}
		}
		
	}
	
	public String toString() {
		StringBuilder sb = new StringBuilder();
		for (ArrayList<String> dimension : dimensionsToSense.keySet()){
			HashSet<Sense> senses = dimensionsToSense.get(dimension);
			for (String d : dimension) sb.append(d + " + ");
			sb.append(" -->\n");
			for (Sense sense : senses){
				sb.append("\t" + sense.getRawSense() + "\n");
			}
			sb.append("\n");
		}
		return sb.toString();
	}
	
	public String getTopLevelLabel(ArrayList<String> dimensions){
		HashSet<Sense> senses = this.dimensionsToSense.get(dimensions);
		if (senses == null) {
			System.out.println("Invalid dimension combination");
			return "Expansion";
		}
		
		HashSet<String> labelSet = new HashSet<String>();
		for (Sense sense : senses){
			labelSet.add(sense.getTopLevelLabel());
		}
		
		Object[] labels = labelSet.toArray();
		if (labels.length > 1) {
			System.out.println("More than one mapping.");
			for (Object l : labels) System.out.println(l);
			return "Expansion";
		}
		return (String)labels[0];
	}
	
	public int numDimensions() {
		return dimensions.size();
	}
	
	public ArrayList<String> getDimensions() {
		return dimensions;
	}

	public String getFeatureFileName(String experimentName, String dimension, String trainingDir) {
		return trainingDir + "/" + experimentName + "." + mappingName + "." + dimension + ".features";
	}
	
	public String getFeatureFileName(String experimentName, String trainingDir) {
		return trainingDir + "/" + experimentName + ".original_label.features";
	}
	
}
