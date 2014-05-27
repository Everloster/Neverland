/**
 * ExecutorStatParserForOld.java
 * com.nearme.base.stat.parse
 *
 * Function： TODO
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2013-4-11 		80036381
 *
 * Copyright (c) 2013 NearMe, All Rights Reserved.
*/

package com.nearme.base.stat.parse;

import java.io.File;
import java.io.IOException;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;

import com.nearme.base.stat.ExecutorStat;
import com.oppo.base.common.NumericUtil;
import com.oppo.base.time.TimeFormat;

/**
 * ExecutorStatParserForOld <br>
 * 抽象执行时间统计文件解析 <br>
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2013-4-11  下午1:50:40
 */
public abstract class AbstractExecutorStatParser {

	protected static final char END_CHAR = '\0';

	private File sourceFile;	//源文件
	private StatInterval statInterval;	//统计间隔
	/**
	 * 外层的key为统计时的identifier，里层的key为日期
	 */
	protected Map<String, Map<String, ExecutorStat>> statMap;

	public AbstractExecutorStatParser() {
		this(null, StatInterval.DAY);
	}

	public AbstractExecutorStatParser(File sourceFile) {
		this(sourceFile, StatInterval.DAY);
	}

	public AbstractExecutorStatParser(File sourceFile, StatInterval statInterval) {
		this.sourceFile = sourceFile;
		this.statInterval = statInterval;
	}

	public void parse() throws IOException {
		if(!sourceFile.exists()) {
			throw new IOException(sourceFile.getAbsolutePath() + " not exist.");
		}

		if(sourceFile.isDirectory()) {
			parseDir(sourceFile);
		} else {
			parseFile(sourceFile);
		}
	}

	public void clearResult() {
		if(null != statMap) {
			statMap.clear();
		}
	}

	/**
	 * 解析目录下所有文件
	 * @param
	 * @return
	 */
	protected void parseDir(File dir) throws IOException {
		File[] subFiles = dir.listFiles();
		for(File file : subFiles) {
			if(file.isDirectory()) {
				parseDir(file);
			} else {
				parseFile(file);
			}
		}
	}

	/**
	 * 解析单个文件
	 * @param
	 * @return
	 */
	protected abstract void parseFile(File file) throws IOException;

	/**
	 * 获取指定标记，指定时间的统计数据,时间格式为yyyyMMddHH、yyyyMMdd或ALL
	 * @param identifier 指定统计的标识
	 * @param time 根据statInterval及time计算出来对应需要被统计的时间
	 * @return
	 */
	public ExecutorStat getExecutorStatInfo(String identifier, String time) {
		if(null == statMap) {
			return null;
		}

		Map<String, ExecutorStat> timeStat = statMap.get(identifier);
		if(null == timeStat) {
			return null;
		} else {
			return timeStat.get(getKey(time));
		}
	}
	
	/**
	 * 根据指定时间的统计数据,时间格式为yyyyMMddHH、yyyyMMdd或ALL
	 * @param time 根据statInterval及time计算出来对应需要被统计的时间
	 * @return
	 */
	public ExecutorStat getExecutorStatInfo(String time) {
		if(null == statMap) {
			return null;
		}
		
		ExecutorStat es = new ExecutorStat("ALL");
		String key = getKey(time);
		for (Map<String, ExecutorStat> timeStat : statMap.values()) {
			ExecutorStat stat = timeStat.get(key);
			if (null != stat) {
				es.setStartTime(Math.min(es.getStartTime(), stat.getStartTime()));
				es.setOutputTime(Math.max(es.getOutputTime(), stat.getOutputTime()));
				es.setExecutorTimes(es.getExecutorTimes() + stat.getExecutorTimes());
				es.setSuccessTimes(es.getSuccessTimes() + stat.getSuccessTimes());
				es.setMaxSuccessTime(Math.max(es.getMaxSuccessTime(), stat.getMaxSuccessTime()));
				es.setMinSuccessTime(Math.min(es.getMinSuccessTime(), stat.getMinSuccessTime()));
				es.setFailTimes(es.getFailTimes() + stat.getFailTimes());
				es.setMaxFailTime(Math.max(es.getMaxFailTime(), stat.getMaxFailTime()));
				es.setMinFailTime(Math.min(es.getMinFailTime(), stat.getMinFailTime()));
			}
		}
		
		return es;
	}

	protected ExecutorStat getExecutorStat(String identifier, String startTime) {
		Map<String, ExecutorStat> timeStat = null;
		if(null == statMap) {
			statMap = new HashMap<String, Map<String, ExecutorStat>>();
		} else {
			timeStat = statMap.get(identifier);
		}

		if(null == timeStat) {
			timeStat = new HashMap<String, ExecutorStat>();
			statMap.put(identifier, timeStat);
		}

		String timeKey = getKey(startTime);
		if(null == timeKey) {
			return null;
		}

		ExecutorStat es = timeStat.get(timeKey);
		if(null == es) {
			es = new ExecutorStat(identifier);
			timeStat.put(timeKey, es);
		}

		return es;
	}

	protected String getKey(String startTime) {
		String format = (startTime.length() == TimeFormat.DEFAULT_FORMAT.length() ?
				TimeFormat.DEFAULT_FORMAT : TimeFormat.DEFAULT_FORMAT.substring(0, startTime.length()));
		Date date = TimeFormat.getDateFromString(startTime, format);

		switch(statInterval) {
		case HOUR:
			return TimeFormat.getTimeString(date, "yyyyMMddHH");
		case DAY:
			return TimeFormat.getTimeString(date, "yyyyMMdd");
		case ALL:
			return "ALL";
		default:
			return null;
		}
	}

	/**
	 * 从行中获取指定信息
	 * @param
	 * @return
	 */
	protected String getInfoFromLine(String line, String startString, char endChar) {
		int startIndex = line.indexOf(startString);
		if(startIndex == -1) {
			return null;
		}

		int infoStartIndex = startIndex + startString.length();
		if(endChar == END_CHAR) {
			return line.substring(infoStartIndex);
		} else {
			int endIndex = line.indexOf(endChar, infoStartIndex);
			return line.substring(infoStartIndex, endIndex);
		}
	}

	/**
	 * 从行中获取指定信息
	 * @param
	 * @return
	 */
	protected int getIntFromLine(String line, String startString, char endChar) {
		String info = getInfoFromLine(line, startString, endChar);
		if(null != info) {
			return NumericUtil.parseInt(info, 0);
		}

		return 0;
	}

//	/**
//	 * 从行中获取指定信息
//	 * @param
//	 * @return
//	 */
//	private double getDoubleFromLine(String line, String startString, char endChar) {
//		String info = getInfoFromLine(line, startString, endChar);
//		if(null != info) {
//			return NumericUtil.parseDouble(info, 0);
//		}
//
//		return 0;
//	}

	/**
	 * 获取解析的源文件（单个文件或目录）
	 * @return  the sourceFile
	 * @since   Ver 1.0
	 */
	public File getSourceFile() {
		return sourceFile;
	}

	/**
	 * 设置解析的源文件（单个文件或目录）
	 * @param   sourceFile
	 * @since   Ver 1.0
	 */
	public void setSourceFile(File sourceFile) {
		this.sourceFile = sourceFile;
	}

	/**
	 * 获取统计间隔
	 * @return  the statInterval
	 * @since   Ver 1.0
	 */
	public StatInterval getStatInterval() {
		return statInterval;
	}

	/**
	 * 设置统计间隔
	 * @param   statInterval
	 * @since   Ver 1.0
	 */
	public void setStatInterval(StatInterval statInterval) {
		this.statInterval = statInterval;
	}
}

