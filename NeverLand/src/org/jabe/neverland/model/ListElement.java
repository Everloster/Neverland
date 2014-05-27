package org.jabe.neverland.model;

import java.util.List;

import android.view.View;
import android.view.ViewGroup;

public interface ListElement<T> {
	public int getLayoutId();
	/**
	 * 把主干逻辑都抽离在此方法
	 * @param position
	 * @param view
	 * @param parent
	 * @return
	 */
	public View getView(int position, View view, ViewGroup parent);
	public boolean isClickable();
	public void click(View view);
	public List<T> getDataList();
	public void remove(int position);
	public void add(T data);
	public void insert(T data, int position);
	public boolean useConvertView();
	public ListElement<T> clone();
	public boolean canReuse();
	public void refreshHolder(IViewHolder holder, int position);
	public IViewHolder getViewHolder();
}
