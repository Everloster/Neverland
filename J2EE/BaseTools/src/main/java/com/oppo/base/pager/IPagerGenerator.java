/**
 * IPagerGenerator.java
 * com.oppo.base.pager
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2011-9-5 		80036381
 *
 * Copyright (c) 2011 NearMe, All Rights Reserved.
*/

package com.oppo.base.pager;
/**
 * ClassName:IPagerGenerator
 * Function: TODO ADD FUNCTION
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-9-5  上午09:25:43
 */
public interface IPagerGenerator {
	public static final String PAGER_REPLACE = "#pager#";
	
	/**
	 * 根据页码号生成页码的html代码，其中pageNumber从1开始
	 * @param pageNumber	页码数
	 * @param pagerType		页码类型
	 * @return
	 */
	String generate(int pageNumber, PagerType pagerType);
}

