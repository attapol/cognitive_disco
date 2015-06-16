package types;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.json.JSONException;
import org.json.JSONObject;

public class DimensionMapper {

	private JSONObject mapping;
	private ArrayList<String> dimensions = new ArrayList<String>();
	private String mappingName;
	
	
	public DimensionMapper(String mappingJson) throws IOException, JSONException{
		String jsonString = FileUtils.readFileToString(new File(mappingJson));
		mapping = new JSONObject(jsonString);
		mappingName = FilenameUtils.getBaseName(mappingJson);
		
		Iterator<String> it = mapping.keys();
		String key = it.next();
		it = mapping.getJSONObject(key).keys();
		while (it.hasNext()) dimensions.add(it.next());
		
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
	
}
