/**
 * ExecutorStatParser.java
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

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

import com.nearme.base.stat.ExecutorStat;
import com.nearme.base.stat.ExecutorStatPool;
import com.oppo.base.common.StringUtil;
import com.oppo.base.file.FileOperate;

/**
 * ClassName:ExecutorStatParser <br>
 * Function: 执行时间统计文件解析 <br>
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2013-4-11  下午1:50:40
 */
public class ExecutorStatParser extends AbstractExecutorStatParser {

	public ExecutorStatParser() {
		super(null, StatInterval.DAY);
	}

	public ExecutorStatParser(File sourceFile) {
		super(sourceFile, StatInterval.DAY);
	}

	public ExecutorStatParser(File sourceFile, StatInterval statInterval) {
		super(sourceFile, statInterval);
	}

	/**
	 * 解析单个文件
	 * @param
	 * @return
	 */
	@Override
	protected void parseFile(File file) throws IOException {
		BufferedReader br = null;

		try {
			br = new BufferedReader(new FileReader(file));

			//按行读取
			String line;
			String startTimeString = null;
			while(null != (line = br.readLine())) {
				if (line.startsWith(ExecutorStatPool.STAT_POOL_START)) {
					startTimeString = getInfoFromLine(line, ExecutorStatPool.STAT_POOL_START, '=');
					// 解析时间
					continue;
				}
				
				//
				String[] secs = StringUtil.split(line, ExecutorStat.SEC_SPLIT_CHAR);
				if (secs.length < 4) {
					continue;
				}
				
				//标记
				String identifier = secs[0];
				ExecutorStat stat = getExecutorStat(
						identifier.substring(0, identifier.length() - 1), 
						startTimeString);
				//总数据
				int execTimes = getIntFromLine(secs[1], ExecutorStat.STAT_EXEC_TIMES, ExecutorStat.COLUMN_SPLIT_CHAR);
				//double execAvg = getDoubleFromLine(line, ExecutorStat.STAT_EXEC_AVG, END_CHAR);
				stat.setExecutorTimes(stat.getExecutorTimes() + execTimes);
				
				//成功数据
				String succ = secs[2];
				int succTimes = getIntFromLine(succ, ExecutorStat.STAT_SUCC_TIMES, ExecutorStat.COLUMN_SPLIT_CHAR);
				//double succAvg = getDoubleFromLine(line, ExecutorStat.STAT_SUCC_AVG, ExecutorStat.COLUMN_SPLIT_CHAR);
				int succMax = getIntFromLine(succ, ExecutorStat.STAT_SUCC_MAX, ExecutorStat.COLUMN_SPLIT_CHAR);
				int succMin = getIntFromLine(succ, ExecutorStat.STAT_SUCC_MIN, END_CHAR);

				stat.setSuccessTimes(stat.getSuccessTimes() + succTimes);
				stat.setMaxSuccessTime(Math.max(succMax, stat.getMaxSuccessTime()));
				stat.setMinSuccessTime(Math.min(succMin, stat.getMinSuccessTime()));
				
				//失败数据
				int failTimes = getIntFromLine(line, ExecutorStat.STAT_FAIL_TIMES, ExecutorStat.COLUMN_SPLIT_CHAR);
				//double failAvg = getDoubleFromLine(line, ExecutorStat.STAT_FAIL_AVG, ExecutorStat.COLUMN_SPLIT_CHAR);
				int failMax = getIntFromLine(line, ExecutorStat.STAT_FAIL_MAX, ExecutorStat.COLUMN_SPLIT_CHAR);
				int failMin = getIntFromLine(line, ExecutorStat.STAT_FAIL_MIN, END_CHAR);

				stat.setFailTimes(stat.getFailTimes() + failTimes);
				stat.setMaxFailTime(Math.max(failMax, stat.getMaxFailTime()));
				stat.setMinFailTime(Math.min(failMin, stat.getMinFailTime()));
			}
		} finally {
			FileOperate.close(br);
		}
	}

	/**
	 * Function Description here
	 * @param
	 * @return
	 * @throws IOException
	 */

	public static void main(String[] args) throws IOException {
		ExecutorStatParser esp = new ExecutorStatParser(new File("h:\\market_xmlapiv1_execstat.log"), StatInterval.HOUR);
		//esp.setStatInterval(StatInterval.HOUR);
		esp.parse();

		for (int i = 0; i < 24; i++) {
			ExecutorStat e1 = esp.getExecutorStatInfo("2013-08-15 " + (i < 10 ? "0" + String.valueOf(i) : String.valueOf(i)));
			System.out.println(e1.toStatString(false, false));
		}
	}
}

