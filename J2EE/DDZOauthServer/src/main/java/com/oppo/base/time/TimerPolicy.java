package com.oppo.base.time;

import java.util.Calendar;

import com.oppo.base.common.NumericUtil;
import com.oppo.base.common.StringUtil;
/**
 * 时间验证策略，主要用于执行定时任务时的时间判断
 * <!--分钟(1-60) 小时(1-24) 天(1-31) 月份(1-12) 一周的第几天(周日为0)-->
 * 例1：* * * * *   每分钟运行
 * 例2： 0 2/* * * * 每两个小时运行
 * 例3： 0 0 3/* * * 每3天的0点0分运行
 * 例4： 0 0 1 2/* * 每两个月的1号0点0分运行
 * 例5： 0 0 1 * 2   每周二的1号0点0分运行
 * 
 * @author yangbo
 * 2011-1-7
 * 
 */
public class TimerPolicy {
	/**
	 * 通配符
	 */
	private final static String WILDCARD = "*"; 
	
	private final static char MULTI_SPLIT = '|';

	private final static char SINGLE_SPLIT = '/';
	
	/**
	 * 判断指定格式在规定时间是否可以执行
	 * @param time 时间(毫秒)
	 * @param timePolicy 格式
	 * @return
	 */
    public static boolean canStart(long time, String timePolicy) {
    	Calendar cal = Calendar.getInstance();
    	cal.setTimeInMillis(time);
    	
        String[] timeSec = StringUtil.split(timePolicy, ' ');
        if (timeSec.length != 5) {
            return false;
        }

        //判断周
        int weekDay = cal.get(Calendar.DAY_OF_WEEK);
        if (!isMultiTimeFit(timeSec[4], weekDay, weekDay, false)) {
            return false;
        }

        //判断月
        int month = cal.get(Calendar.MONTH);
        if (!isMultiTimeFit(timeSec[3], month, month, true)) {
            return false;
        }

        //判断月中的日
        int yearDay = cal.get(Calendar.DAY_OF_YEAR);
        int monthDay = cal.get(Calendar.DAY_OF_MONTH);
        if (!isMultiTimeFit(timeSec[2], yearDay, monthDay, true)) {
            return false;
        }

        //判断小时,加上月的小时数,即每月开始时重新计算
        int dayHour = cal.get(Calendar.HOUR_OF_DAY);
        int monthHour = cal.get(Calendar.DAY_OF_MONTH) * 24 + dayHour;
        if (!isMultiTimeFit(timeSec[1], monthHour, dayHour, true)) {
            return false;
        }

        //判断分,加上小时的分钟数,即每天开始时重新计算
        int hourMinute = cal.get(Calendar.MINUTE);
        int dayMinute = cal.get(Calendar.HOUR_OF_DAY) * 60 + hourMinute;
        if (!isMultiTimeFit(timeSec[0], dayMinute, hourMinute, true)) {
            return false;
        }

        return true;
    }
    
    /**
	 * 判断某一复合节点是否符合规则
	 * 该算法还需要完善
	 * @param sec 节点值
	 * @param timeLong 需要匹配的值
	 * @param timeShort 需要精确匹配的值
	 * @param needRepeat 当前节点是否支持循环，暂不支持周
	 * @return
	 */
    private static boolean isMultiTimeFit(String sec, int timeLong, int timeShort, boolean needRepeat) {
    	String[] secs = StringUtil.split(sec, MULTI_SPLIT);
    	for(int i = 0; i < secs.length; i++) {
    		//逐个节点的匹配
    		boolean isSingleFit = isTimeFit(secs[i], timeLong, timeShort, needRepeat);
    		//其中一个节点匹配则匹配
    		if(isSingleFit) {
    			return true;
    		}
    	}
    	
    	return false;
    }
	
	/**
	 * 判断某一单节点是否符合规则
	 * 该算法还需要完善
	 * @param sec 节点值
	 * @param timeLong 需要匹配的值
	 * @param timeShort 需要精确匹配的值
	 * @param needRepeat 当前节点是否支持循环，暂不支持周
	 * @return
	 */
    private static boolean isTimeFit(String sec, int timeLong, int timeShort, boolean needRepeat) {
        String[] secs = StringUtil.split(sec, SINGLE_SPLIT);
        if (secs.length == 1) {
            if (sec.equals(WILDCARD)) { //为*则匹配
                return true;
            }

            int aTime = NumericUtil.parseInt(sec, -1);
            if (aTime == timeShort) {//与当前值相等则匹配
                return true;
            } else {
                return false;
            }
        } else if (secs.length == 2) {
            if (!secs[1].equals(WILDCARD)) {//后一个必须是*
            
                return false;
            }
            if (secs[0].equals(WILDCARD)) {//第一个是*则匹配所有
                return true;
            }

            int aTime = NumericUtil.parseInt(secs[0], -1);
            if (aTime > 0 && needRepeat && (timeLong % aTime == 0)) {
                return true;
            }
        }

        return false;
    }
    
    public static void main(String[] args) {
    	boolean isFit = TimerPolicy.canStart(TimeFormat.getDateFromString("2012-05-12 02:00:00").getTime(), "10|00 02 * * *");
    	System.out.print(isFit);
    }
}
