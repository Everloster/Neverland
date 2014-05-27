package com.oppo.base.pager;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

import com.oppo.base.common.StringUtil;

/**
 * ClassName:Pager Function: 提供分页控件功能
 * 
 * @author 80053851
 * @version
 * @since Ver 1.0
 * @Date 2011-8-30 上午10:02:04
 */
public class Pager {

	private int total; // 总记录数
	private int currentPage;// 当前页数
	private int pageSize; // 每页记录数
	private int pageNumber; // 页面显示的页码个数
	private IPagerDesc pagerDesc;
	private IPagerGenerator pagerGenerator;

	private String PAGE_FIRST = "首页";// FirstPage
	private String PAGE_UP = "上一页";// FrontPage
	private String PAGE_DOWN = "下一页";// NextPage
	private String PAGE_LAST = "尾页";// LastPage

	private String PAGE_ALL = "共";// all
	private String PAGE_RECORD = "条记录";// "records";
	private String PAGE_PER = "每页";
	private String PAGE_PER_PAGE = "条";// per page
	private String PAGE_CURRENT = "当前";// current
	private String PAGE_CURRENT_PAGE = "页";// page

	private String NO_DATA_FOUND = "没有找到数据";// page

	public Pager() {
		this(0);
	}
	
	public Pager(int language) {

		if (language == 1) {
			initEN();
		}

		pagerDesc = new DefaultPagerDesc();
		pagerGenerator = new DefaultPagerGenerator();
	}

	private void initEN() {
		PAGE_FIRST = "FirstPage";
		PAGE_UP = "FrontPage";// FrontPage
		PAGE_DOWN = "NextPage";//
		PAGE_LAST = "LastPage";//

		PAGE_ALL = "All";// all
		PAGE_RECORD = "records";// "records";
		PAGE_PER = "";
		PAGE_PER_PAGE = "per page";// per page
		PAGE_CURRENT = "current";// current
		PAGE_CURRENT_PAGE = "page";// page

		NO_DATA_FOUND = "data not found...";
	}

	/**
	 * 获取总记录数
	 * 
	 * @return the total
	 * @since Ver 1.0
	 */
	public int getTotal() {
		return total;
	}

	/**
	 * 设置总记录数
	 * 
	 * @param total
	 * @since Ver 1.0
	 */
	public void setTotal(int total) {
		this.total = total;
	}

	/**
	 * 获取当前页数
	 * 
	 * @return the currentPage
	 * @since Ver 1.0
	 */
	public int getCurrentPage() {
		return currentPage;
	}

	/**
	 * 设置当前页数
	 * 
	 * @param currentPage
	 * @since Ver 1.0
	 */
	public void setCurrentPage(int currentPage) {
		this.currentPage = currentPage;
	}

	/**
	 * 获取每页记录数
	 * 
	 * @return the pageSize
	 * @since Ver 1.0
	 */
	public int getPageSize() {
		return pageSize;
	}

	/**
	 * 设置每页记录数
	 * 
	 * @param pageSize
	 * @since Ver 1.0
	 */
	public void setPageSize(int pageSize) {
		this.pageSize = pageSize;
	}

	/**
	 * 获取页面显示的页码个数
	 * 
	 * @return the pageNumber
	 * @since Ver 1.0
	 */
	public int getPageNumber() {
		return pageNumber;
	}

	/**
	 * 设置页面显示的页码个数
	 * 
	 * @param pageNumber
	 * @since Ver 1.0
	 */
	public void setPageNumber(int pageNumber) {
		this.pageNumber = pageNumber;
	}

	/**
	 * 获取pagerGenerator
	 * 
	 * @return the pagerGenerator
	 * @since Ver 1.0
	 */
	public IPagerDesc getPagerDesc() {
		return pagerDesc;
	}

	/**
	 * 设置pagerGenerator
	 * 
	 * @param pagerGenerator
	 * @since Ver 1.0
	 */
	public void setPagerDesc(IPagerDesc pagerDesc) {
		this.pagerDesc = pagerDesc;
	}

	/**
	 * 获取pagerGenerator
	 * 
	 * @return the pagerGenerator
	 * @since Ver 1.0
	 */
	public IPagerGenerator getPagerGenerator() {
		return pagerGenerator;
	}

	/**
	 * 设置pagerGenerator
	 * 
	 * @param pagerGenerator
	 * @since Ver 1.0
	 */
	public void setPagerGenerator(IPagerGenerator pagerGenerator) {
		this.pagerGenerator = pagerGenerator;
	}

	/**
	 * 获取翻页代码
	 * 
	 * @param total
	 *            总记录数
	 * @param currentPage
	 *            当前页
	 * @param pageSize
	 *            每页记录数
	 * @param pageNumber
	 *            每页页码显示数
	 * @return
	 */
	public String getPager(int total, int currentPage, int pageSize,
			int pageNumber) {
		this.setTotal(total);
		this.setCurrentPage(currentPage);
		this.setPageSize(pageSize);
		this.setPageNumber(pageNumber);

		return getPager();
	}

	public String getPager() {
		StringBuffer pagerBuffer = new StringBuffer();
		// 前缀部分
		pagerBuffer.append(pagerDesc.getPagerTablePrefix());

		if (total != 0) { // 如果总记录数>0则进行输出
			// 计算总页数
			int totalPage = total / pageSize;
			if (total % pageSize != 0) {
				++totalPage;
			}

			// 计算需要显示的页码范围
			int startPage = currentPage - pageNumber / 2; // 起始页

			if (startPage <= 1) {
				startPage = 1;
			}

			int endPage = startPage + pageNumber - 1; // 结束页
			if (endPage >= totalPage) {
				endPage = totalPage;

				// 修正起始页
				startPage = endPage - pageNumber;
				if (startPage <= 1) {
					startPage = 1;
				}
			}

			int prevPage = currentPage - 1; // 上一页
			int nextPage = currentPage + 1;

			// 分页描述
			pagerBuffer.append(pagerDesc.getPagerDesc(totalPage, total,
					pageSize, currentPage));
			pagerBuffer.append(pagerDesc.getPagerRowPrefix());

			// 上一页
			if (prevPage > 0) {
				pagerBuffer.append(pagerGenerator.generate(prevPage,
						PagerType.PREVIOUS));
				pagerBuffer.append("&nbsp;");
			}
			// 如果开始页不是第一页则进行输出
			if (startPage != 1) {
				pagerBuffer.append(pagerGenerator.generate(1, PagerType.BEGIN));
				pagerBuffer.append("&nbsp;");
			}

			// 循环输出所有页
			for (int i = startPage; i <= endPage; i++) {
				pagerBuffer.append(pagerGenerator.generate(i,
						(i == this.currentPage) ? PagerType.CURRENT
								: PagerType.NORMAL));
				pagerBuffer.append("&nbsp;");
			}

			// 如果结束页不是最后一页则进行输出
			if (endPage != totalPage) {
				pagerBuffer.append(pagerGenerator.generate(totalPage,
						PagerType.END));
				pagerBuffer.append("&nbsp;");
			}
			// 下一页
			if (nextPage <= totalPage) {
				pagerBuffer.append(pagerGenerator.generate(nextPage,
						PagerType.NEXT));
				pagerBuffer.append("&nbsp;");
			}

			pagerBuffer.append(pagerDesc.getPagerRowSuffix());
		} else { // 无记录
			pagerBuffer.append(pagerDesc.getNoPagerDesc());
		}

		// 后缀部分
		pagerBuffer.append(pagerDesc.getPagerTableSuffix());
		return pagerBuffer.toString();
	}

	/**
	 * 生成分页控件表
	 * 
	 * @param total
	 *            总记录数
	 * @param currentPage
	 *            当前显示页数
	 * @param pageSize
	 *            页面大小
	 * @param length
	 *            页码显示长度
	 * @param action
	 *            action请求
	 * @param requestarams
	 *            请求参数表
	 * @param iconMap
	 *            首页、上一页、下一页、尾页 图标
	 * @return 分页控件表串，可以返回空串
	 */
	@SuppressWarnings("rawtypes")
	public String getPager(int total, int currentPage, int pageSize,
			int length, String action, HashMap<String, String> requestMap,
			HashMap<String, String> iconMap) {

		StringBuffer sb = new StringBuffer();
		sb.append("<table width='100%' class='pager_table_style'><tbody><tr>");
		if (StringUtil.isNullOrEmpty(action)) {
			// 请求为空时返回空
			return null;
		}
		action = action.trim();

		StringBuffer ub = new StringBuffer();
		ub.append(action);
		ub.append("?");
		if (requestMap != null) {
			// 组装请求URL
			Iterator it = requestMap.entrySet().iterator();
			while (it.hasNext()) {
				Map.Entry entry = (Map.Entry) it.next();
				Object key = entry.getKey();
				Object value = entry.getValue();
				if (key != null && value != null) {
					ub.append(key.toString());
					ub.append("=");
					ub.append(value.toString());
					ub.append("&");
				}
			}
		}
		if (total != 0) {
			// 生成js请求事件
			ub.append("pageSize=");
			ub.append(pageSize);
			ub.append("&");
			sb.append("<script type='text/javascript'>");
			sb.append("function pageRedirect(obj){");
			ub.append("currentPage=");
			sb.append("var url='" + ub.toString()
					+ "'+obj+'&sm='+Math.random();");
			sb.append("window.location.href=url;");
			sb.append("}");
			sb.append("</script>");
			int totalPage = ((total - 1) / pageSize) + 1;
			int front = 1;
			int next = totalPage;
			if (currentPage > 1) {
				front = currentPage - 1;
			}
			if (currentPage < totalPage) {
				next = currentPage + 1;
			}
			// 分页描述信息
			sb.append("<td align='left'  class='pager_td_style'>");
			sb.append(PAGE_ALL);
			sb.append("<font color='red'>");
			sb.append(total);
			sb.append("</font>");
			sb.append(this.PAGE_RECORD);
			sb.append("&nbsp;");
			sb.append(this.PAGE_PER);
			sb.append("<font color='red'>");
			sb.append(pageSize);
			sb.append("</font>");
			sb.append(this.PAGE_PER_PAGE);
			sb.append("&nbsp;");
			sb.append(this.PAGE_CURRENT);
			sb.append("<font color='red'>");
			sb.append(currentPage);
			sb.append("/");
			sb.append(totalPage);
			sb.append("</font>");
			sb.append(this.PAGE_CURRENT_PAGE);
			sb.append("</td>");
			// 分页事件
			sb.append("<td align='right' class='pager_td_style'>");
			sb.append("<a href='#' onClick='pageRedirect(1)'>");
			String img1 = null;
			if (iconMap != null) {
				img1 = this.getIconString(PagerConstants.PAGE_FIRST, iconMap);
			}
			if (img1 != null) {
				sb.append(img1);
			} else {
				sb.append(this.PAGE_FIRST);
			}
			sb.append("</a>");
			sb.append("&nbsp;");
			sb.append("<a href='#' onClick='pageRedirect(" + front + ")'>");
			String img2 = null;
			if (iconMap != null) {
				img2 = this.getIconString(PagerConstants.PAGE_UP, iconMap);
			}
			if (img2 != null) {
				sb.append(img2);
			} else {
				sb.append(this.PAGE_UP);
			}
			sb.append("</a>");
			sb.append("&nbsp;&nbsp;");
			if (totalPage > length) {
				if (currentPage + length > totalPage) {
					sb.append("<a href='#' onClick='pageRedirect(1)'>1...</a>");
					sb.append("&nbsp;");
					for (int k1 = (totalPage - length); k1 < totalPage + 1; k1++) {
						sb.append("<a href='#' onClick='pageRedirect(" + k1
								+ ")'>" + k1 + "</a>");
						sb.append("&nbsp;");
					}
				} else if (currentPage - length < 0) {
					for (int k2 = 1; k2 < length + 1; k2++) {
						sb.append("<a href='#' onClick='pageRedirect(" + k2
								+ ")'>" + k2 + "</a>");
						sb.append("&nbsp;");
					}
					sb.append("...<a href='#' onClick='pageRedirect("
							+ totalPage + ")'>" + totalPage + "</a>");
					sb.append("&nbsp;");
				} else {
					int start = currentPage - (length / 2);
					int end = currentPage + (length / 2);
					sb.append("<a href='#' onClick='pageRedirect(1)'>1</a>...");
					sb.append("&nbsp;");
					for (int k3 = start; k3 <= end; k3++) {
						sb.append("<a href='#' onClick='pageRedirect(" + k3
								+ ")'>" + k3 + "</a>");
						sb.append("&nbsp;");
					}
					sb.append("...<a href='#' onClick='pageRedirect("
							+ totalPage + ")'>" + totalPage + "</a>");
					sb.append("&nbsp;");
				}
			} else {
				for (int k = 0; k < totalPage; k++) {
					sb.append("<a href='#' onClick='pageRedirect(" + (k + 1)
							+ ")'>" + (k + 1) + "</a>");
					sb.append("&nbsp;");
				}
			}
			sb.append("&nbsp;");
			sb.append("<a href='#' onClick='pageRedirect(" + next + ")'>");
			String img3 = null;
			if (iconMap != null) {
				img3 = this.getIconString(PagerConstants.PAGE_DOWN, iconMap);
			}
			if (img3 != null) {
				sb.append(img3);
			} else {
				sb.append(this.PAGE_DOWN);
			}
			sb.append("</a>");
			sb.append("&nbsp;");
			sb.append("<a href='#' onClick='pageRedirect(" + totalPage + ")'>");
			String img4 = null;
			if (iconMap != null) {
				img4 = this.getIconString(PagerConstants.PAGE_LAST, iconMap);
			}
			if (img4 != null) {
				sb.append(img4);
			} else {
				sb.append(this.PAGE_LAST);
			}
			sb.append("</a>");
			sb.append("</td>");

		} else {
			// 没有数据
			sb.append("<td>" + NO_DATA_FOUND + "</td>");
		}
		sb.append("</tr></tbody></table>");

		return sb.toString();
	}

	/**
	 * 生成分页控件表
	 * 
	 * @param total
	 *            总记录数
	 * @param currentPage
	 *            当前显示页数
	 * @param pageSize
	 *            页面大小
	 * @param length
	 *            页码显示长度
	 * @param action
	 *            action请求
	 * @param requestarams
	 *            请求参数表
	 * @return 分页控件表串，可以返回空串
	 */
	public String getPager(int total, int currentPage, int pageSize,
			int length, String action, HashMap<String, String> requestParams) {
		return this.getPager(total, currentPage, pageSize, length, action,
				requestParams, null);
	}

	private String getIconString(String key, HashMap<String, String> map) {
		if (map != null) {
			if (map.containsKey(key)) {
				String src = map.get(key);
				if (src != null && "".compareTo(src) != 0) {
					src = src.trim();
					String istr = "<img src='" + src
							+ "' width='43' height='15'/>";
					return istr;
				}
			}
		}
		return null;
	}

	public static void main(String[] args) {
		Pager p = new Pager(1);
		String str = p.getJsPager(100, 20, 5, 10);
		System.out.println(str);
		// DefaultPagerGenerator pagerGenerator = new DefaultPagerGenerator();
		// 方式一：url翻页方式
		// pagerGenerator
		// .setUrlPattern("http://appstore.nearme.com.cn/a.jsp?page="
		// + IPagerGenerator.PAGER_REPLACE);
		// 方式二：js翻页方式
		// pagerGenerator.setHrefAdditional("onClick='goto(" +
		// IPagerGenerator.PAGER_REPLACE + ");'");
		// p.setPagerGenerator(pagerGenerator);
		// String html = p.getPager(181, 8, 10, 10);
		// try {
		// FileOperate.saveContentToFile("F:\\a.html", html, "gbk");
		// } catch (Exception e) {
		// e.printStackTrace();
		// }
	}

	public String getJsPager(int total, int currentPage, int pageSize,
			int length) {

		StringBuffer sb = new StringBuffer();
		sb.append("<table width='100%' class='pager_table_style'><tbody><tr>");
		if (total != 0) {
			// 生成js请求事件
			int totalPage = ((total - 1) / pageSize) + 1;
			int front = 1;
			int next = totalPage;
			if (currentPage > 1) {
				front = currentPage - 1;
			}
			if (currentPage < totalPage) {
				next = currentPage + 1;
			}
			// 分页描述信息
			sb.append("<td align='left'  class='pager_td_style'>");
			sb.append(this.PAGE_ALL + "<font color='red'>");
			sb.append(total);
			sb.append("</font>" + this.PAGE_RECORD + "&nbsp;" + this.PAGE_PER
					+ "<font color='red'>");
			sb.append(pageSize);
			sb.append("</font>" + this.PAGE_PER_PAGE + "&nbsp;"
					+ this.PAGE_CURRENT + "<font color='red'>");
			sb.append(currentPage);
			sb.append("/");
			sb.append(totalPage);
			sb.append("</font>" + this.PAGE_CURRENT_PAGE);
			sb.append("</td>");
			// 分页事件
			sb.append("<td align='right' class='pager_td_style'>");
			sb.append("<a href='#' onClick='pageRedirect(1)'>");
			sb.append(this.PAGE_FIRST);
			sb.append("</a>");
			sb.append("&nbsp;");
			sb.append("<a href='#' onClick='pageRedirect(" + front + ")'>");
			sb.append(this.PAGE_UP);
			sb.append("</a>");
			sb.append("&nbsp;&nbsp;");
			if (totalPage > length) {
				if (currentPage + length > totalPage) {
					sb.append("<a href='#' onClick='pageRedirect(1)'>1</a>...");
					sb.append("&nbsp;");
					for (int k1 = (totalPage - length); k1 < totalPage + 1; k1++) {
						sb.append("<a href='#' onClick='pageRedirect(" + k1
								+ ")'>" + k1 + "</a>");
						sb.append("&nbsp;");
					}
				} else if (currentPage - length < 0) {
					for (int k2 = 1; k2 < length + 1; k2++) {
						sb.append("<a href='#' onClick='pageRedirect(" + k2
								+ ")'>" + k2 + "</a>");
						sb.append("&nbsp;");
					}
					sb.append("...<a href='#' onClick='pageRedirect("
							+ totalPage + ")'>" + totalPage + "</a>");
					sb.append("&nbsp;");
				} else {
					int start = currentPage - (length / 2);
					int end = currentPage + (length / 2);
					sb.append("...<a href='#' onClick='pageRedirect(1)'>1</a>");
					sb.append("&nbsp;");
					for (int k3 = start; k3 <= end; k3++) {
						sb.append("<a href='#' onClick='pageRedirect(" + k3
								+ ")'>" + k3 + "</a>");
						sb.append("&nbsp;");
					}
					sb.append("...<a href='#' onClick='pageRedirect("
							+ totalPage + ")'>" + totalPage + "</a>");
					sb.append("&nbsp;");
				}
			} else {
				for (int k = 0; k < totalPage; k++) {
					sb.append("<a href='#' onClick='pageRedirect(" + (k + 1)
							+ ")'>" + (k + 1) + "</a>");
					sb.append("&nbsp;");
				}
			}
			sb.append("&nbsp;");
			sb.append("<a href='#' onClick='pageRedirect(" + next + ")'>");
			sb.append(this.PAGE_DOWN);
			sb.append("</a>");
			sb.append("&nbsp;");
			sb.append("<a href='#' onClick='pageRedirect(" + totalPage + ")'>");
			sb.append(this.PAGE_LAST);
			sb.append("</a>");
			sb.append("</td>");

		} else {
			// 没有数据
			sb.append("<td>" + this.NO_DATA_FOUND + "</td>");
		}
		sb.append("</tr></tbody></table>");

		return sb.toString();
	}
}
