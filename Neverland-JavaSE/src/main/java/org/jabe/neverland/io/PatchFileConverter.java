package org.jabe.neverland.io;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class PatchFileConverter {

	public static void main(String[] args) {

		final File rootPath = new File(ROOT_PATH);

		convertDir(rootPath);
	}

	private static void convertFile(File file) {
		if (file.isDirectory()) {
			convertDir(file);
		} else {
			convertRealFile(file);
		}
	}

	private static void convertDir(File file) {
		if (file.isDirectory()) {
			final File rootPath = file;
			final String[] childsFileName = rootPath.list();
			for (String string : childsFileName) {
				convertFile(new File(rootPath, string));
			}
		} else {
			convertRealFile(file);
		}
	}
	
	private static volatile boolean needChange = false;

	private static void convertRealFile(File file) {
		if (file.isFile()) {
			if (file.getName().endsWith(".java")) {
				try {
					final File newFile = new File(file.getCanonicalPath() + ".txt");
					newFile.delete();
					newFile.createNewFile();
					final BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(newFile));
					final BufferedReader bufferIo = new BufferedReader(new FileReader(file));
					String line = "";
					while ((line = bufferIo.readLine()) != null) {
						if (line.equals("import com.nearme.gamecenter.open.R;")) {
							needChange = true;
							continue;
						}
						bufferedWriter.write(convertStringLine(line));
						bufferedWriter.newLine();
					}
					bufferIo.close();
					bufferedWriter.close();
					if (needChange) {
						String fileName = file.getCanonicalPath();
						file.delete();
						newFile.renameTo(new File(fileName));
					} else {
						newFile.delete();
					}
					needChange = false;
				} catch (FileNotFoundException e) {
					e.printStackTrace();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
	}
	
	private static String PATTERN = "[R][.][a-zA-Z]+[.][a-z_0-9A-Z]+";
	
	private static String convertStringLine(String line) {
		final Pattern pattern = Pattern.compile(PATTERN);
		final Matcher matcher = pattern.matcher(line);
		String newLine = new String(line);
		while(matcher.find()) {
			needChange = true;
			final String group = matcher.group();
			final String con = convertString(group);
			newLine = newLine.replaceFirst(group, con);
			
		}
		return  newLine;
	}
	
	private static String convertString(String content) {
		final String data = content;
		final String[] list = data.split("\\.");
		String out = "";
		if (list[1].equals("id")) {
			out = "GetResource.getIdResource(\"" + list[2] + "\")";
		} else if (list[1].equals("layout")) {
			out = "GetResource.getLayoutResource(\"" + list[2] + "\")";
		} else if (list[1].equals("drawable")) {
			out = "GetResource.getDrawableResource(\"" + list[2] + "\")";
		} else if (list[1].equals("string")) {
			out = "GetResource.getStringResource(\"" + list[2] + "\")";
		} else if (list[1].equals("style")) {
			out = "GetResource.getStyleResource(\"" + list[2] + "\")";
		} else if (list[1].equals("dimen")) {
			out = "GetResource.getDimenResource(\"" + list[2] + "\")";
		} else if (list[1].equals("anim")) {
			out = "GetResource.getAnimResource(\"" + list[2] + "\")";
		} else if (list[1].equals("styleable")) {
			out = "GetResource.getStyleableResource(\"" + list[2] + "\")";
		}
		if (out.equals("")) {
			return content;
		}
		return out;
	}

	private static final String ROOT_PATH = "E:\\WorkspaceForEclipseNew\\OpenSDK1.0\\src";
}
