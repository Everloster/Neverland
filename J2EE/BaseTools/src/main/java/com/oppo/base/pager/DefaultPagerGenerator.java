/**
 * DefaultPagerGenerator.java
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

import com.oppo.base.common.OConstants;
import com.oppo.base.common.StringUtil;

/**
 * ClassName:DefaultPagerGenerator
 * Function: TODO ADD FUNCTION
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-9-5  上午09:26:45
 */
public class DefaultPagerGenerator implements IPagerGenerator {
	private String urlPattern;
	private String hrefAdditional;

	public DefaultPagerGenerator() {
		
	}
	
	public DefaultPagerGenerator(String urlPattern) {
		this.urlPattern = urlPattern;
	}
	
	/**
	 * 获取a标签的url链接Pattern，如http://nearme.com.cn/test.jsp?page=#pager#,没有则为空
	 * @return  the urlPattern
	 * @since   Ver 1.0
	 */
	public String getUrlPattern() {
		return urlPattern;
	}

	/**
	 * 设置a标签的url链接Pattern，如http://nearme.com.cn/test.jsp?page=#pager#,没有可设为#
	 * @param   urlPattern    
	 * @since   Ver 1.0
	 */
	public void setUrlPattern(String urlPattern) {
		this.urlPattern = urlPattern;
	}

	/**
	 * 获取a标签href以外的属性，如onclick='go(#pager#'),没有则为空
	 * @return  the hrefAdditional
	 * @since   Ver 1.0
	 */
	public String getHrefAdditional() {
		return hrefAdditional;
	}

	/**
	 * 设置a标签href以外的属性，如onclick='go(#pager#'),没有则留空
	 * @param   hrefAdditional    
	 * @since   Ver 1.0
	 */
	public void setHrefAdditional(String hrefAdditional) {
		this.hrefAdditional = hrefAdditional;
	}

	/**
	 * 根据页码及其类型，生成对应的链接
	 * @see com.oppo.base.pager.IPagerGenerator#generate(int, com.oppo.base.pager.PagerType)
	 */
	@Override
	public String generate(int pageNumber, PagerType pagerType) {
		StringBuffer pagerHtmlBuffer = new StringBuffer();
		switch(pagerType) {
		case BEGIN:
			pagerHtmlBuffer.append(getATag(pageNumber, "首页", pagerType));
			break;
		case END:
			pagerHtmlBuffer.append(getATag(pageNumber, "尾页", pagerType));
			break;
		case PREVIOUS:
			pagerHtmlBuffer.append(getATag(pageNumber, "上一页", pagerType));
			break;
		case NEXT:
			pagerHtmlBuffer.append(getATag(pageNumber, "下一页", pagerType));
			break;
		case CURRENT:
			pagerHtmlBuffer.append(getATag(-1, String.valueOf(pageNumber), pagerType));
			break;	
		case NORMAL:
			pagerHtmlBuffer.append(getATag(pageNumber, String.valueOf(pageNumber), pagerType));
			break;
		}
		
		return pagerHtmlBuffer.toString();
	}
	
	protected String getATag(int pageNumber, String tagValue, PagerType pagerType) {
		if(pagerType == PagerType.CURRENT) {
			return String.valueOf(tagValue);
		} else {
			StringBuffer tagBuffer = new StringBuffer();
			tagBuffer.append("<a href='");
			tagBuffer.append(getUrlView(pageNumber));
			tagBuffer.append("' ");
			tagBuffer.append(getHrefAdditionalView(pageNumber));
			tagBuffer.append(" >");
			tagBuffer.append(tagValue);
			tagBuffer.append("</a>");
			
			return tagBuffer.toString();
		}
	}
	
	/**
	 * 获取指定页的url
	 * @param 
	 * @return
	 */
	protected String getUrlView(int pageNumber) {
		if(pageNumber < 1) {
			return "#";
		} 
		
		String pattern =  this.getUrlPattern();
		if(null == pattern) {
			return "#";
		}
		
		return this.getUrlPattern().replace(IPagerGenerator.PAGER_REPLACE, String.valueOf(pageNumber));
	}
	
	/**
	 * 获取指定页的a标签href外的属性
	 * @param 
	 * @return
	 */
	protected String getHrefAdditionalView(int pageNumber) {
		String hAdd = this.getHrefAdditional();
		if(StringUtil.isNullOrEmpty(hAdd)) {
			return OConstants.EMPTY_STRING;
		} else {
			return hAdd.replace(IPagerGenerator.PAGER_REPLACE, String.valueOf(pageNumber));
		}
	}
}

