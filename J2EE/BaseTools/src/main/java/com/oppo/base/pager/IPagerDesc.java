/**
 * IPageGenerator.java
 * com.oppo.base.pager
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2011-9-2 		80036381
 *
 * Copyright (c) 2011 NearMe, All Rights Reserved.
*/

package com.oppo.base.pager;
/**
 * ClassName:IPageGenerator
 * Function: 页码信息产生器
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-9-2  下午01:57:44
 */
public interface IPagerDesc {	
	/**
	 * 获取分页内容的页面设计前缀部分
	 * @param 
	 * @return
	 */
	String getPagerTablePrefix();
	
	String getPagerRowPrefix();
	
	/**
	 * 获取分页内容的页面设计后缀部分
	 * @param 
	 * @return
	 */
	String getPagerTableSuffix();
	
	String getPagerRowSuffix();
	
	/**
	 * 获取分页描述信息
	 * @param totalPage		总页数
	 * @param totalRecord	总记录数
	 * @param pageSize		每页记录数
	 * @param currentPage	当前页码
	 * @return
	 */
	String getPagerDesc(int totalPage, int totalRecord, int pageSize, int currentPage);
	
	/**
	 * 无数据时的显示
	 * @param 
	 * @return
	 */
	String getNoPagerDesc();
}

