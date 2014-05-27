package org.jabe.neverland.model;

import android.view.View;

public interface IViewHolder {

	/**
	 * save the view's all reference
	 * @param view
	 * @param position
	 */
	public void saveView(View view, int position, boolean save);
}
