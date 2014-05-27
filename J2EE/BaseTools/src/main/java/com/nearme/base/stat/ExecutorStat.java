/**
 * ExecutorStat.java
 * com.nearme.base.stat
 *
 * Function： TODO
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2011-12-12 		80036381
 *
 * Copyright (c) 2011 NearMe, All Rights Reserved.
*/

package com.nearme.base.stat;

import java.io.Serializable;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.locks.ReentrantLock;

/**
 * ClassName:ExecutorStat <br>
 * Function: 程序执行时间、成功率等数据的统计 <br>
 * 每一个实体对应程序中执行的某一个统计点,多个统计点之间互不干涉 <br>
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2011-12-12  上午09:22:13
 */
public class ExecutorStat implements Serializable {
	/**
	 * @since Ver 1.1
	 */
	private static final long serialVersionUID = 1L;

	public static final String STAT_IDENTIFIER = "";
	public static final String STAT_START = "start at ";
	public static final String STAT_OUTPUT = "output at ";
	public static final String STAT_EXEC_TIMES = "tttl:";
	public static final String STAT_EXEC_AVG = "tavg:";
	public static final String STAT_SUCC_TIMES = "sttl:";
	public static final String STAT_SUCC_AVG = "savg:";
	public static final String STAT_SUCC_MAX = "smax:";
	public static final String STAT_SUCC_MIN = "smin:";
	public static final String STAT_FAIL_TIMES = "fttl:";
	public static final String STAT_FAIL_AVG = "favg:";
	public static final String STAT_FAIL_MAX = "fmax:";
	public static final String STAT_FAIL_MIN = "fmin:";
	public static final char COLUMN_SPLIT_CHAR = ' ';
	public static final char SEC_SPLIT_CHAR = '|';
	public static final String CHILD_START = "\n{\n";
	public static final String CHILD_END = "\n}\n";

	private String identifier;		//当前统计项的标识
	private final int childNodeSize;		//子节点个数
	private List<ExecutorStat> childNodeList;	//子节点list
	private long startTime;			//统计开始时间
	private long outputTime;		//统计输出时间

	private int executorTimes;		//执行次数
	private long totalExecutorTime;	//总执行时间，以毫秒为单位
	private double avgExecutorTime;	//平均执行时间

	private int successTimes;		//执行成功次数
	private long totalSuccessTime;	//总成功时间，以毫秒为单位
	private double avgSuccessTime;	//执行成功的平均时间，以毫秒为单位
	private long maxSuccessTime;	//执行成功的最大时间，以毫秒为单位
	private long minSuccessTime;	//执行成功的最小时间，以毫秒为单位

	private int failTimes;			//执行失败次数
	private long totalFailTime;		//总失败时间，以毫秒为单位
	private double avgFailTime;		//执行失败的平均时间，以毫秒为单位
	private long maxFailTime;		//执行失败的最大时间，以毫秒为单位
	private long minFailTime;		//执行失败的最小时间，以毫秒为单位

	private boolean hasStatInfo;		//是否加入了统计信息

	private ReentrantLock lock = new ReentrantLock();//同步锁
	private static NumberFormat DECIMAL_FORMAT = new DecimalFormat("#.###");

	/**
	 *
	 * Creates a new instance of ExecutorStat.
	 *
	 * @param identifier 统计节点标识
	 */
	public ExecutorStat(String identifier) {
		this(identifier, 0);
	}

	/**
	 *
	 * Creates a new instance of ExecutorStat.
	 *
	 * @param identifier 统计节点标识
	 * @param childNodeSize 子节点个数
	 */
	public ExecutorStat(String identifier, int childNodeSize) {
		if(null == identifier) {
			throw new NullPointerException("identifier can not be null.");
		}

		this.identifier = identifier;
		this.childNodeSize = childNodeSize;
		this.changeChildSize(this.childNodeSize);

		reset();
	}

	//修改子节点数目
	private void changeChildSize(int size) {
		if(size > 0) {
			//如果数目未变则不重新设置
			if(null != childNodeList && size == childNodeList.size()) {
				return;
			}

			childNodeList = new ArrayList<ExecutorStat>(size);
			for(int i = 0; i < size; i++) {
				//字节点命名规则为 父节点‘identifier_子节点位置’,如 'identifier_0'
				childNodeList.add(new ExecutorStat(this.identifier + "_" + size));
			}
		} else {
			childNodeList = null;
		}
	}

	/**
	 * 添加子节点统计信息
	 * @param index 子节点位置，从0开始，该值不能大于等于初始设置的childNodeSize
	 * @param isSuccess	是否执行成功
	 * @param executorTime	执行时间
	 * @return
	 */
	public void addChildStat(int index, boolean isSuccess, long executorTime) {
		if(index < 0) {
			return;
		}

		if(index >= this.childNodeSize) {
			throw new IndexOutOfBoundsException("child size is " + childNodeSize + ", input index too large: " + index);
		}

		this.childNodeList.get(index).addStat(isSuccess, executorTime);
		hasStatInfo = true;
	}

	/**
	 * 添加一个统计数据
	 * @param isSuccess	是否执行成功
	 * @param executeTime	执行时间
	 * @return
	 */
	public void addStat(boolean isSuccess, long executeTime) {
		lock.lock();
		try {
			//执行次数，总时间及平均执行时间
			executorTimes++;
			totalExecutorTime += executeTime;
			avgExecutorTime = totalExecutorTime * 1.0 / executorTimes;

			if(isSuccess) {
				//成功
				successTimes++;
				totalSuccessTime += executeTime;
				avgSuccessTime = totalSuccessTime * 1.0 / successTimes;

				//计算最大时间
				maxSuccessTime = Math.max(executeTime, maxSuccessTime);
				//最小时间
				minSuccessTime = Math.min(executeTime, minSuccessTime);
			} else {
				//失败
				failTimes++;
				totalFailTime += executeTime;
				avgFailTime = totalFailTime * 1.0 / failTimes;

				//最大时间
				maxFailTime = Math.max(executeTime, maxFailTime);
				//最小时间
				minFailTime = Math.min(executeTime, minFailTime);
			}

			hasStatInfo = true;
		} finally {
			lock.unlock();
		}
	}

	/**
	 * 设置开始时间（在分析统计日志等情景使用）
	 * @param
	 * @return
	 */
	public void setStartTime(long startTime) {
		this.startTime = startTime;
	}

	public long getStartTime() {
		return this.startTime;
	}

	/**
	 * 设置输出时间（在分析统计日志等情景使用）
	 * @param
	 * @return
	 */
	public void setOutputTime(long outputTime) {
		this.outputTime = outputTime;
	}

	public long getOutputTime() {
		return this.outputTime;
	}

	/**
	 * 获得统计信息的文字格式
	 * @param
	 * @return
	 */
	public String toStatString(boolean reset) {
		return toStatString(true, reset);
	}

	/**
	 * 获得统计信息的文字格式
	 * @param
	 * @return
	 */
	public String toStatString(boolean emptyJudge, boolean reset) {
		if(emptyJudge) {
			if(this.hasStatInfo()) {
				outputTime = System.currentTimeMillis();
			} else {
				return null;
			}
		}

		StringBuilder statBuilder = new StringBuilder(256 * (1 + childNodeSize));
		//输出标记及开始结束时间
		statBuilder.append(identifier).append('\t');
		
		//输出总信息
		statBuilder.append(SEC_SPLIT_CHAR);
		statBuilder.append(STAT_EXEC_TIMES).append(executorTimes).append(COLUMN_SPLIT_CHAR);//总次数
		statBuilder.append(STAT_EXEC_AVG).append(getFormatNumber(avgExecutorTime)).append(COLUMN_SPLIT_CHAR);//总平均时间

		//输出成功信息
		//statBuilder.append(COLUMN_SPLIT_CHAR);
		statBuilder.append(SEC_SPLIT_CHAR);
		statBuilder.append(STAT_SUCC_TIMES).append(successTimes).append(COLUMN_SPLIT_CHAR);//成功次数
		statBuilder.append(STAT_SUCC_AVG).append(getFormatNumber(avgSuccessTime)).append(COLUMN_SPLIT_CHAR);//成功平均时间
		statBuilder.append(STAT_SUCC_MAX).append(maxSuccessTime).append(COLUMN_SPLIT_CHAR);	//最大成功执行时间
		statBuilder.append(STAT_SUCC_MIN).append(minSuccessTime).append(COLUMN_SPLIT_CHAR);	//最小成功执行时间

		//输出失败信息
		//statBuilder.append(COLUMN_SPLIT_CHAR);
		statBuilder.append(SEC_SPLIT_CHAR);
		statBuilder.append(STAT_FAIL_TIMES).append(failTimes).append(COLUMN_SPLIT_CHAR);//失败次数
		statBuilder.append(STAT_FAIL_AVG).append(getFormatNumber(avgFailTime)).append(COLUMN_SPLIT_CHAR);//失败平均时间
		statBuilder.append(STAT_FAIL_MAX).append(maxFailTime).append(COLUMN_SPLIT_CHAR);	//最大失败执行时间
		statBuilder.append(STAT_FAIL_MIN).append(minFailTime);	//最小失败执行时间
		
		//输出子节点
		if(this.childNodeSize > 0) {
			statBuilder.append(CHILD_START);
			for(int i = 0; i < this.childNodeSize; i++) {
				String statInfo = this.childNodeList.get(i).toStatString(emptyJudge, reset);
				if(null != statInfo) {
					statBuilder.append(statInfo);
				}
			}

			statBuilder.append(CHILD_END);
		}
		
//		if (!emptyJudge) {
//			statBuilder.append(STAT_START).append(TimeFormat.getTimeString(startTime)).append(COLUMN_SPLIT_CHAR);
//			statBuilder.append(STAT_OUTPUT).append(TimeFormat.getTimeString(outputTime));
//		}

		//重置数据
		if(reset) {
			reset();
		}

		return statBuilder.toString();
	}

	private String getFormatNumber(double number) {
		return DECIMAL_FORMAT.format(number);
	}

	/**
	 * 重置数据
	 * @param
	 * @return
	 */
	public void reset() {
		lock.lock();

		try {
			startTime = System.currentTimeMillis();	//统计开始
			avgExecutorTime = 0;	//平均执行时间
			totalExecutorTime = 0;	//总执行时间，以毫秒为单位
			executorTimes = 0;		//执行次数

			successTimes = 0;		//执行成功次数
			totalSuccessTime = 0;	//总成功时间，以毫秒为单位
			avgSuccessTime = 0;		//执行成功的平均时间
			maxSuccessTime = 0;		//执行成功的最大时间
			minSuccessTime = 0;		//执行成功的最小时间

			failTimes = 0;			//执行失败次数
			totalFailTime = 0;		//总失败时间，以毫秒为单位
			avgFailTime = 0;		//执行失败的平均时间
			maxFailTime = 0;		//执行失败的最大时间
			minFailTime = 0;		//执行失败的最小时间

			hasStatInfo = false;
		} finally {
			lock.unlock();
		}
	}

	/**
	 * 获取当前统计项的标识
	 * @return  the identifier
	 * @since   Ver 1.0
	 */
	public String getIdentifier() {
		return identifier;
	}
	/**
	 * 设置当前统计项的标识
	 * @param   identifier
	 * @since   Ver 1.0
	 */
	public void setIdentifier(String identifier) {
		this.identifier = identifier;
	}
	/**
	 * 获取平均执行时间，以毫秒为单位
	 * @return  the avgExecutorTime
	 * @since   Ver 1.0
	 */
	public double getAvgExecutorTime() {
		return avgExecutorTime;
	}
	/**
	 * 设置平均执行时间，以毫秒为单位
	 * @param   avgExecutorTime
	 * @since   Ver 1.0
	 */
	public void setAvgExecutorTime(double avgExecutorTime) {
		this.avgExecutorTime = avgExecutorTime;
	}
	/**
	 * 获取执行次数
	 * @return  the executorTimes
	 * @since   Ver 1.0
	 */
	public int getExecutorTimes() {
		return executorTimes;
	}
	/**
	 * 设置执行次数
	 * @param   executorTimes
	 * @since   Ver 1.0
	 */
	public void setExecutorTimes(int executorTimes) {
		this.executorTimes = executorTimes;
	}
	/**
	 * 获取执行成功次数
	 * @return  the successTimes
	 * @since   Ver 1.0
	 */
	public int getSuccessTimes() {
		return successTimes;
	}
	/**
	 * 设置执行成功次数
	 * @param   successTimes
	 * @since   Ver 1.0
	 */
	public void setSuccessTimes(int successTimes) {
		this.successTimes = successTimes;
	}
	/**
	 * 获取执行成功的平均时间，以毫秒为单位
	 * @return  the avgSuccessTime
	 * @since   Ver 1.0
	 */
	public double getAvgSuccessTime() {
		return avgSuccessTime;
	}
	/**
	 * 设置执行成功的平均时间，以毫秒为单位
	 * @param   avgSuccessTime
	 * @since   Ver 1.0
	 */
	public void setAvgSuccessTime(double avgSuccessTime) {
		this.avgSuccessTime = avgSuccessTime;
	}
	/**
	 * 获取执行成功的最大时间，以毫秒为单位
	 * @return  the maxSuccessTime
	 * @since   Ver 1.0
	 */
	public long getMaxSuccessTime() {
		return maxSuccessTime;
	}
	/**
	 * 设置执行成功的最大时间，以毫秒为单位
	 * @param   maxSuccessTime
	 * @since   Ver 1.0
	 */
	public void setMaxSuccessTime(long maxSuccessTime) {
		this.maxSuccessTime = maxSuccessTime;
	}
	/**
	 * 获取执行成功的最小时间，以毫秒为单位
	 * @return  the minSuccessTime
	 * @since   Ver 1.0
	 */
	public long getMinSuccessTime() {
		return minSuccessTime;
	}
	/**
	 * 设置执行成功的最小时间，以毫秒为单位
	 * @param   minSuccessTime
	 * @since   Ver 1.0
	 */
	public void setMinSuccessTime(long minSuccessTime) {
		this.minSuccessTime = minSuccessTime;
	}
	/**
	 * 获取执行失败次数
	 * @return  the failTimes
	 * @since   Ver 1.0
	 */
	public int getFailTimes() {
		return failTimes;
	}
	/**
	 * 设置执行失败次数
	 * @param   failTimes
	 * @since   Ver 1.0
	 */
	public void setFailTimes(int failTimes) {
		this.failTimes = failTimes;
	}
	/**
	 * 获取执行失败的平均时间，以毫秒为单位
	 * @return  the avgFailTime
	 * @since   Ver 1.0
	 */
	public double getAvgFailTime() {
		return avgFailTime;
	}
	/**
	 * 设置执行失败的平均时间，以毫秒为单位
	 * @param   avgFailTime
	 * @since   Ver 1.0
	 */
	public void setAvgFailTime(double avgFailTime) {
		this.avgFailTime = avgFailTime;
	}
	/**
	 * 获取执行失败的最大时间，以毫秒为单位
	 * @return  the maxFailTime
	 * @since   Ver 1.0
	 */
	public long getMaxFailTime() {
		return maxFailTime;
	}
	/**
	 * 设置执行失败的最大时间，以毫秒为单位
	 * @param   maxFailTime
	 * @since   Ver 1.0
	 */
	public void setMaxFailTime(long maxFailTime) {
		this.maxFailTime = maxFailTime;
	}
	/**
	 * 获取执行失败的最小时间，以毫秒为单位
	 * @return  the minFailTime
	 * @since   Ver 1.0
	 */
	public long getMinFailTime() {
		return minFailTime;
	}
	/**
	 * 设置执行失败的最小时间，以毫秒为单位
	 * @param   minFailTime
	 * @since   Ver 1.0
	 */
	public void setMinFailTime(long minFailTime) {
		this.minFailTime = minFailTime;
	}
	/**
	 * 获取总执行时间，以毫秒为单位
	 * @return  the totalExecutorTime
	 * @since   Ver 1.0
	 */
	public long getTotalExecutorTime() {
		return totalExecutorTime;
	}

	/**
	 * 设置总执行时间，以毫秒为单位
	 * @param   totalExecutorTime
	 * @since   Ver 1.0
	 */
	public void setTotalExecutorTime(long totalExecutorTime) {
		this.totalExecutorTime = totalExecutorTime;
	}

	/**
	 * 获取总成功时间，以毫秒为单位
	 * @return  the totalSuccessTime
	 * @since   Ver 1.0
	 */
	public long getTotalSuccessTime() {
		return totalSuccessTime;
	}

	/**
	 * 设置总成功时间，以毫秒为单位
	 * @param   totalSuccessTime
	 * @since   Ver 1.0
	 */
	public void setTotalSuccessTime(long totalSuccessTime) {
		this.totalSuccessTime = totalSuccessTime;
	}

	/**
	 * 获取总失败时间，以毫秒为单位
	 * @return  the totalFailTime
	 * @since   Ver 1.0
	 */
	public long getTotalFailTime() {
		return totalFailTime;
	}

	/**
	 * 设置总失败时间，以毫秒为单位
	 * @param   totalFailTime
	 * @since   Ver 1.0
	 */
	public void setTotalFailTime(long totalFailTime) {
		this.totalFailTime = totalFailTime;
	}

	/**
	 * 获取是否有加入的统计信息
	 * @return  the hasStat
	 * @since   Ver 1.0
	 */
	public boolean hasStatInfo() {
		return hasStatInfo;
	}

	public static void main(String[] args) {
		ExecutorStat es = new ExecutorStat("my test",1);
		es.addStat(true, 1000);
		es.addStat(true, 1023);
		es.addStat(true, 1010);
		es.addStat(true, 1020);
//		es.addStat(false, 1111);
		es.addChildStat(0, true, 1000);
		es.addChildStat(0, true, 1030);
		es.addChildStat(0, false, 1000);
		System.out.println(es.toStatString(true));
	}
}

