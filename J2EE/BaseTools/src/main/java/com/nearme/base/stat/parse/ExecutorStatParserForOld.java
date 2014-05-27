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

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.sql.Timestamp;

import com.nearme.base.stat.ExecutorStat;
import com.oppo.base.file.FileOperate;
import com.oppo.base.time.TimeFormat;

/**
 * ExecutorStatParserForOld <br>
 * 老版本执行时间统计文件解析 <br>
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2013-4-11  下午1:50:40
 */
public class ExecutorStatParserForOld extends AbstractExecutorStatParser {

	public ExecutorStatParserForOld() {
		super(null, StatInterval.DAY);
	}

	public ExecutorStatParserForOld(File sourceFile) {
		super(sourceFile, StatInterval.DAY);
	}

	public ExecutorStatParserForOld(File sourceFile, StatInterval statInterval) {
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
			ParseStep step = ParseStep.INIT; //单个统计解析到的行数(从标记行开始)
			ExecutorStat lastStat = null;
			while(null != (line = br.readLine())) {
				switch(step) {
				case INIT://还未开始解析或前一次数据解析完
					String identifier = getInfoFromLine(line, "identifier:", '\t');
					if(null == identifier) {
						step = ParseStep.INIT;
						continue;
					}
					String startTimeString = getInfoFromLine(line, "start at ", '\t');

					lastStat = this.getExecutorStat(identifier, startTimeString);

					//设置开始时间
					long startTime = Math.min(Timestamp.valueOf(startTimeString).getTime(), lastStat.getStartTime());
					lastStat.setStartTime(startTime);
					//设置输出时间
					String outputTimeString = getInfoFromLine(line, "output at ", END_CHAR);
					long outputTime = Math.max(Timestamp.valueOf(outputTimeString).getTime(), lastStat.getOutputTime());
					lastStat.setOutputTime(outputTime);

					step = ParseStep.TOTAL;
					break;
				case TOTAL: //第一行,总数据
					int execTimes = getIntFromLine(line, ExecutorStat.STAT_EXEC_TIMES, '\t');
					//double execAvg = getDoubleFromLine(line, ExecutorStat.STAT_EXEC_AVG, END_CHAR);

					lastStat.setExecutorTimes(lastStat.getExecutorTimes() + execTimes);
					step = ParseStep.SUCC;
					break;
				case SUCC: //第二行,成功数据
					int succTimes = getIntFromLine(line, ExecutorStat.STAT_SUCC_TIMES, '\t');
					//double succAvg = getDoubleFromLine(line, ExecutorStat.STAT_SUCC_AVG, ExecutorStat.COLUMN_SPLIT_CHAR);
					int succMax = getIntFromLine(line, ExecutorStat.STAT_SUCC_MAX, '\t');
					int succMin = getIntFromLine(line, ExecutorStat.STAT_SUCC_MIN, END_CHAR);

					lastStat.setSuccessTimes(lastStat.getSuccessTimes() + succTimes);
					lastStat.setMaxSuccessTime(Math.max(succMax, lastStat.getMaxSuccessTime()));
					lastStat.setMinSuccessTime(Math.min(succMin, lastStat.getMinSuccessTime()));

					step = ParseStep.FAIL;
					break;
				case FAIL: //第三号，失败数据
					int failTimes = getIntFromLine(line, ExecutorStat.STAT_FAIL_TIMES, '\t');
					//double failAvg = getDoubleFromLine(line, ExecutorStat.STAT_FAIL_AVG, ExecutorStat.COLUMN_SPLIT_CHAR);
					int failMax = getIntFromLine(line, ExecutorStat.STAT_FAIL_MAX, '\t');
					int failMin = getIntFromLine(line, ExecutorStat.STAT_FAIL_MIN, END_CHAR);

					lastStat.setFailTimes(lastStat.getFailTimes() + failTimes);
					lastStat.setMaxFailTime(Math.max(failMax, lastStat.getMaxFailTime()));
					lastStat.setMinFailTime(Math.min(failMin, lastStat.getMinFailTime()));

					step = ParseStep.INIT;
					break;
				default:
					step = ParseStep.INIT;
					break;
				}
			}
		} finally {
			FileOperate.close(br);
		}
	}

	enum ParseStep {
		INIT,
		TOTAL,
		SUCC,
		FAIL
	}

	/**
	 * Function Description here
	 * @param
	 * @return
	 * @throws IOException
	 */

	public static void main(String[] args) throws IOException {
		ExecutorStatParserForOld esp = new ExecutorStatParserForOld(new File("h:\\market_core_execstat_2013-07-26.0.log"), StatInterval.ALL);
		esp.setStatInterval(StatInterval.HOUR);
		esp.parse();

		ExecutorStat e0 = esp.getExecutorStatInfo(
				"com.nearme.market.core.product.upgrade.CheckUpgradeHandler",
				"2013-08-06 21");
		System.out.println(e0.toStatString(false, false));
		
		for (int i = 0; i < 24; i++) {
			ExecutorStat e1 = esp.getExecutorStatInfo("2013-08-06 " + (i < 10 ? "0" + String.valueOf(i) : String.valueOf(i)));
			System.out.println(TimeFormat.getTimeString(e1.getStartTime()) + " " + e1.toStatString(false, false));
		}
	}
}

