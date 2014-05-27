/**
 * PagerDescription.java
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
 * ClassName:PagerDescription
 * Function: 分页控件html描述,如果要改变控件外观，请重写此类
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-9-5  上午09:12:28
 */
public class DefaultPagerDesc implements IPagerDesc {
	/**
	 * 获取分页内容的页面设计前缀部分
	 * @param 
	 * @return
	 */
	public String getPagerTablePrefix() {
		return "<table class='pager_table_style'><tbody><tr>";
	}
	
	public String getPagerRowPrefix() {
		return "<td class='pager_td_style'>";
	}
	
	/**
	 * 获取分页内容的页面设计后缀部分
	 * @param 
	 * @return
	 */
	public String getPagerTableSuffix() {
		return "</tr></tbody></table>";
	}
	
	public String getPagerRowSuffix() {
		return "</td>";
	}
	
	/**
	 * 获取分页描述信息
	 * @param 
	 * @return
	 */
	public String getPagerDesc(int totalPage, int totalRecord, int pageSize, int currentPage) {
		StringBuilder descBuffer = new StringBuilder();
		// 分页描述信息
		descBuffer.append("<td class='pager_td_style'>");
		descBuffer.append("共<font class='pagerTotal'>");
		descBuffer.append(totalRecord);
		descBuffer.append("</font>条记录&nbsp;每页<font class='pagerSize'>");
		descBuffer.append(pageSize);
		descBuffer.append("</font>条&nbsp;当前<font class='pagerCurrent'>");
		descBuffer.append(currentPage);
		descBuffer.append("/");
		descBuffer.append(totalPage);
		descBuffer.append("</font>页");
		descBuffer.append("</td>");
		
		return descBuffer.toString();
	}
	
	
	/**
	 * 无数据时的显示
	 * @param 
	 * @return
	 */
	public String getNoPagerDesc() {
		return "<td class='pagerNoData'>没有找到数据</td>";
	}
}

