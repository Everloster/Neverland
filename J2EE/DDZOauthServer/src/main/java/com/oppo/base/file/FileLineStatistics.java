/**
 * FileLineStatistics.java
 * com.oppo.base.file
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2011-10-13 		80036381
 *
 * Copyright (c) 2011 NearMe, All Rights Reserved.
*/

package com.oppo.base.file;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.HashMap;
import java.util.Map;

import com.oppo.base.common.OConstants;

/**
 * ClassName:FileLineStatistics
 * Function: 文件行数统计
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-10-13  下午04:51:58
 */
public class FileLineStatistics {
	private File statisticsFile;
	private boolean needEmpty;
	private boolean needRemark;
	private String[] includeFileFormat;
	private String[] excludeFileFormat;
	
	public FileLineStatistics() {
	}
	
	public FileLineStatistics(String filePath) {
		this(new File(filePath));
	}
	
	public FileLineStatistics(File file) {
		this.statisticsFile = file;
	}
	
	/**
	 * 统计出各个文件的行数
	 * @param 
	 * @return
	 */
	public Map<File, Long> statistic() throws Exception {
		Map<File, Long> statMap = new HashMap<File, Long>();
		
		if(null == statisticsFile || !statisticsFile.exists()) {
			return statMap;
		}
		
		if(statisticsFile.isDirectory()) {
			statMap = statisticForDir(statisticsFile);
		} else {
			long total = statisticForFile(statisticsFile);
			if(total >= 0) {
				statMap.put(statisticsFile, total);
			}
		}
		
		return statMap;
	}
	
	/**
	 * 统计出指定目录下各个文件的行数
	 * @param 
	 * @return
	 */
	public Map<File, Long> statisticForDir(File dir) throws Exception {
		Map<File, Long> statMap = new HashMap<File, Long>();
		
		File[] files = dir.listFiles();
		for(int i = 0; i < files.length; i++) {
			File f = files[i];
			if(f.isDirectory()) {
				Map<File, Long> subStatMap = statisticForDir(f);
				statMap.putAll(subStatMap);
			} else {
				long total = statisticForFile(f);
				if(total >= 0) {
					statMap.put(f, total);
				}
			}
		}
		
		return statMap;
	}
	
	/**
	 * 获取文件总行数
	 * @return
	 * @throws Exception 
	 */
	public long statisticForFile(File file) throws Exception {
		if(!file.exists()) {
			return -1;
		}
		
		String filePath = file.getAbsolutePath().toLowerCase();
		//查看是否为排除的文件
		if(null != excludeFileFormat) {
			for(int i = 0; i < excludeFileFormat.length; i++) {
				if(filePath.endsWith(excludeFileFormat[i].toLowerCase())) {
					return -1;
				}
			}
		}
		
		//查看是否为排除
		if(null != includeFileFormat) {
			boolean contains = false;
			for(int i = 0; i < includeFileFormat.length; i++) {
				if(filePath.endsWith(includeFileFormat[i].toLowerCase())) {
					contains = true;
					break;
				}
			}
			
			if(!contains) {
				return -1;
			}
		}
		
		long total = 0;
		BufferedReader lineReader = null;
		try {
			lineReader = new BufferedReader(new FileReader(file));
			
			String line;
			//是否是多行注释
			boolean isRemark = false; 
			while(null != (line = lineReader.readLine())) {
				line = line.trim();
				boolean isEmptyLine = OConstants.EMPTY_STRING.equals(line);

				//不需要空行且为空则跳过
				if(!needEmpty && isEmptyLine) {
					continue;
				}

				//如果是多行注释且未结束则为注释，然后判断是否为单行注释
				boolean isRemarkLine = isRemark ? isRemark : (line.startsWith("//"));
				if(isRemark && line.endsWith("*/")) {
					isRemark = false;
				}
				
				if(!needRemark && isRemarkLine) {
					continue;
				}
				
				if(!isRemark && line.startsWith("/*")) {
					isRemark = true;
				}
				
				total++;
			}
		} finally {
			FileOperate.close(lineReader);
		}
		
		return total;
	}
	
	/**
	 * 获取statisticsFile
	 * @return  the statisticsFile
	 * @since   Ver 1.0
	 */
	public File getStatisticsFile() {
		return statisticsFile;
	}

	/**
	 * 设置statisticsFile
	 * @param   statisticsFile    
	 * @since   Ver 1.0
	 */
	public void setStatisticsFile(File statisticsFile) {
		this.statisticsFile = statisticsFile;
	}

	/**
	 * 获取needEmpty
	 * @return  the needEmpty
	 * @since   Ver 1.0
	 */
	public boolean isNeedEmpty() {
		return needEmpty;
	}

	/**
	 * 设置needEmpty
	 * @param   needEmpty    
	 * @since   Ver 1.0
	 */
	public void setNeedEmpty(boolean needEmpty) {
		this.needEmpty = needEmpty;
	}

	/**
	 * 获取needRemark
	 * @return  the needRemark
	 * @since   Ver 1.0
	 */
	public boolean isNeedRemark() {
		return needRemark;
	}

	/**
	 * 设置needRemark
	 * @param   needRemark    
	 * @since   Ver 1.0
	 */
	public void setNeedRemark(boolean needRemark) {
		this.needRemark = needRemark;
	}

	/**
	 * 获取includeFileFormat
	 * @return  the includeFileFormat
	 * @since   Ver 1.0
	 */
	public String[] getIncludeFileFormat() {
		return includeFileFormat;
	}

	/**
	 * 设置includeFileFormat
	 * @param   includeFileFormat    
	 * @since   Ver 1.0
	 */
	public void setIncludeFileFormat(String[] includeFileFormat) {
		this.includeFileFormat = includeFileFormat;
	}

	/**
	 * 获取excludeFileFormat
	 * @return  the excludeFileFormat
	 * @since   Ver 1.0
	 */
	public String[] getExcludeFileFormat() {
		return excludeFileFormat;
	}

	/**
	 * 设置excludeFileFormat
	 * @param   excludeFileFormat    
	 * @since   Ver 1.0
	 */
	public void setExcludeFileFormat(String[] excludeFileFormat) {
		this.excludeFileFormat = excludeFileFormat;
	}

	public static void main(String[] args) throws Exception {
		FileLineStatistics flStat = new FileLineStatistics("F:\\OPPOGame\\nearme_game\\gamecenter\\Trunk");
		flStat.setExcludeFileFormat(new String[] {".jar",".svn-base", ".tmp"});
		Map<File, Long> statMap = flStat.statistic();
		System.out.println("Total: " + statMap.size());
		for(File f : statMap.keySet()) {
			long t = statMap.get(f);
			if(t > 300) {
				System.out.println(t + ":\t\t" + f.getAbsolutePath());
			}
		}
	}
}

